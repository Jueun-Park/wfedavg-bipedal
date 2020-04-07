import os
import csv
import numpy as np
import pickle
import gym
import gym_selected_bipedal
import time

from stable_baselines import ACKTR
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import MlpPolicy
from rnd import RandomNetworkDistillation
from train_rnd_callback import rnd_training_dataset, train_rnd_callback
from scipy.special import softmax


env_ids = ["grass", "stump", "stairs", "pit"]

class WeightedFedLearner(object):
    def __init__(self, num_clients, n_cpu, alpha, trial, timesteps, sub_model_timesteps, prefix, test_mode=0):
        self.dir_name = '{}_num_clients_{}'.format(prefix, num_clients)
        if not os.path.isdir('./model/{}'.format(self.dir_name)):
            os.mkdir('./model/{}'.format(self.dir_name))
        self.dir_name = '{}_num_clients_{}/{}'.format(
            prefix, num_clients, trial)
        if not os.path.isdir('./model/{}'.format(self.dir_name)):
            os.mkdir('./model/{}'.format(self.dir_name))
        self.num_sub_envs = len(env_ids)
        self.num_clients = num_clients
        self.n_clients_in_subenv = self.num_clients//self.num_sub_envs
        self.prefix = prefix
        self.n_cpu = n_cpu
        self.alpha = alpha
        self.timesteps = timesteps
        self.sub_model_timesteps = sub_model_timesteps
        self.test_mode = test_mode
        configure = [num_clients, timesteps, sub_model_timesteps]
        with open('./model/{}/log.csv'.format(self.dir_name), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(configure)
        return

    def learn(self, train_rnd_callback):
        # self.default_model_learn()
        for n_client in range(self.n_clients_in_subenv):
            st = time.time()
            for n_env_id in range(self.num_sub_envs):
                self.sub_model_learn(n_client, n_env_id, train_rnd_callback)
                print(f">> n_client {n_client}/{self.n_clients_in_subenv}, n_env_id {n_env_id}/{self.num_sub_envs}")
                print(f">> client training time: {time.time() - st}")
        model = self.model_align()
        return

    def sub_model_learn(self, n_client, n_env_id, train_rnd_callback):
        number = n_client*self.num_sub_envs + n_env_id
        env = SubprocVecEnv([lambda: gym.make(f"selected-bipedal-{env_ids[n_env_id]}-v0") for i in range(self.n_cpu)])
        model = ACKTR.load(f"./model/{env_ids[n_env_id]}/model.zip", env)
        model.verbose = 0
        model.learn(total_timesteps=self.sub_model_timesteps,
                    callback=train_rnd_callback)
        model.save(
            './model/{}/client_policy_{}.zip'.format(self.dir_name, number))
        with open("tmp/rnd_dataset.pkl", "rb") as f:
            rnd_training_dataset = pickle.load(f)
        print(
            f"> rnd training data size: {len(rnd_training_dataset)}")
        rnd = RandomNetworkDistillation(
            env.observation_space.shape[0], use_cuda=True, tensorboard=True)
        rnd.learn(np.array(rnd_training_dataset), n_steps=2000)
        rnd.save(path="./model/{}/rnd_{}/".format(self.dir_name,
                                                  number, subfix=str(number)))
        return model

    def get_weights_from_rnd(self, k=-100):
        num_data = 1000
        weights = np.ndarray((self.num_clients, num_data))
        x_test = []
        env = make_vec_env(f"selected-bipedal-{env_ids[self.test_mode]}-v0", n_envs=1)
        model = ACKTR.load(f"./model/{env_ids[self.test_mode]}/model.zip")
        obs = env.reset()
        for _ in range(num_data):
            action, _ = model.predict(obs)
            obs, _, _, _ = env.step(action)
            x_test.append(obs[0])
        del env, model

        for i in range(self.num_clients):
            rnd = RandomNetworkDistillation(
                input_size=24, use_cuda=True, tensorboard=False)
            rnd.load(path=f"./model/{self.dir_name}/rnd_{i}")
            w = rnd.get_intrinsic_reward(x_test)
            w = np.array(w)
            w = k * w
            weights[i] = w
        weights = np.transpose(weights)
        for j in range(num_data):
            weights[j] = softmax(weights[j])
        weights = np.mean(weights, axis=0)
        return weights

    def model_align(self, weight_scale=100):
        sub_model_parameters = []
        for i in range(self.num_clients):
            sub_model = ACKTR.load(
                './model/{}/client_policy_{}.zip'.format(self.dir_name, i))
            sub_model_parameters.append(sub_model.get_parameters())
        model = ACKTR.load(
            './model/{}/model.zip'.format(env_ids[self.test_mode]))
        parameters = model.get_parameters()
        key = parameters.keys()
        for layer_key in key:
            layer_parameter = []
            for i in range(self.num_clients):
                layer_parameter.append(sub_model_parameters[i][layer_key])
            weights = self.get_weights_from_rnd(-weight_scale)
            # print(f"> DEBUG: {weights}\n{weights.shape}, sum: {np.sum(weights)}")
            delta = np.average(layer_parameter, axis=0, weights=weights)
            parameters[layer_key] = (1 - self.alpha) * \
                parameters[layer_key] + self.alpha * delta
        model._save_to_file(
            './model/{}/fedavg_policy'.format(self.dir_name), params=parameters)
        model.load_parameters(
            './model/{}/fedavg_policy.zip'.format(self.dir_name))
        model.save('./model/{}/fedavg_policy'.format(self.dir_name))
        return model
