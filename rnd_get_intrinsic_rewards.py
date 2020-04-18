import argparse
import numpy as np
import gym
import gym_selected_bipedal
from stable_baselines import ACKTR
from stable_baselines.common import make_vec_env
import csv
from pathlib import Path
import multiprocessing as mp
from itertools import product
import time

from modules.rnd import RandomNetworkDistillation
from info import subenv_dict, seed


num_test = 10000


def get_intrinsic_reward(base_index):
    intrinsic_rewards = [[] for _ in range(len(subenv_dict))]
    # base env
    base_name = subenv_dict[base_index]
    base_env = make_vec_env(
        f"selected-bipedal-{base_name}-v0", n_envs=1, seed=seed)
    base_agent = ACKTR.load(f"./base_agent/{base_name}/model.zip")

    # rnd model
    rnd_dict = {}
    for client_env in subenv_dict.values():
        rnd = RandomNetworkDistillation(input_size=24)
        rnd.load(f"./base{base_index}_client_model/{client_env}/rnd")
        rnd_dict[client_env] = rnd
    obs = base_env.reset()
    for _ in range(num_test):
        for i, client_env in subenv_dict.items():
            intrinsic_rewards[i].append(rnd_dict[client_env].get_intrinsic_reward(obs))
        action = base_agent.predict(obs)
        obs, reward, done, info = base_env.step(action[0])
        if done:
            obs = base_env.reset()
    return intrinsic_rewards


if __name__ == "__main__":
    intrinsic_rewards = []
    start_time = time.time()
    with mp.Pool(None) as pool:
        intrinsic_rewards = pool.map(get_intrinsic_reward, subenv_dict.keys())
    minutes, seconds = divmod(time.time() - start_time, 60)
    print(f"Time processed: {minutes:.0f}m {seconds:.0f}s")

    intrinsic_rewards = np.array(intrinsic_rewards)

    # axis1: base env, axis2: client env
    cri_mean = np.mean(intrinsic_rewards, axis=2)
    std_mean = np.mean(cri_mean, axis=1)
    print(cri_mean)
    print(std_mean)
    tmp = np.std(intrinsic_rewards, axis=2)
    std_std = np.std(tmp, axis=1)
    print(std_std)
    standardized_ir = (cri_mean - std_mean) / std_std
    print(standardized_ir)

    Path("log").mkdir(parents=True, exist_ok=True)
    with open(f"log/rnd_log.csv", "w", newline="") as f:
        wf = csv.writer(f)
        for i in range(4):
            wf.writerow([f"base{i}"] + list(standardized_ir[i]))
