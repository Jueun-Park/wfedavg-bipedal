import argparse
import numpy as np
import gym
import gym_selected_bipedal
from stable_baselines import ACKTR
from stable_baselines.common import make_vec_env
import csv
from pathlib import Path

from modules.rnd import RandomNetworkDistillation
from info import subenv_dict, seed


num_test = 100

if __name__ == "__main__":
    intrinsic_rewards = [[[] for _ in range(len(subenv_dict))] for _ in range(len(subenv_dict))]
    for subenv_i, subenv in subenv_dict.items():
        # data
        test_env = make_vec_env(
            f"selected-bipedal-{subenv}-v0", n_envs=1, seed=seed)
        test_agent = ACKTR.load(f"./model/{subenv}/model.zip")

        model_label = []
        for i in range(len(subenv_dict)):
            # rnd model
            rnd = RandomNetworkDistillation(input_size=24)
            rnd.load(f"./model/bipedal_num_clients_4/0/rnd_{i}")
            obs = test_env.reset()
            for _ in range(num_test):
                intrinsic_rewards[subenv_i][i].append(rnd.get_intrinsic_reward(obs))
                action = test_agent.predict(obs)
                obs, reward, done, info = test_env.step(action[0])
                if done:
                    obs = test_env.reset()

    # axis1: data, axis2: model
    cri_mean = np.mean(intrinsic_rewards, axis=2)
    std_mean = np.mean(cri_mean, axis=1)
    print(cri_mean)
    print(std_mean)
    tmp = np.std(intrinsic_rewards, axis=2)
    std_std = np.std(tmp, axis=1)
    print(std_std)
    standardized_ir = (cri_mean - std_mean) / std_std
    print(standardized_ir)
    
    # Path("log").mkdir(parents=True, exist_ok=True)
    # with open(f"log/rnd_log_base{subenv_i}.csv", "w", newline="") as f:
    #     wf = csv.writer(f)
    #     wf.writerow([f"base{subenv_i}"])
