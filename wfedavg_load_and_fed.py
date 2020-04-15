import gym
import gym_selected_bipedal
from stable_baselines import ACKTR
from stable_baselines.common import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
import numpy as np
import argparse
import csv
from pathlib import Path
import multiprocessing as mp
from itertools import product
import time

from modules.gen_weights import grid_weights_gen
from info import subenv_dict, seed, alpha


def model_align(w, base_parameter_dict, sub_model_parameters, alpha):
    keys = base_parameter_dict.keys()
    for key in keys:
        layer_parameter = []
        for i in range(4):
            layer_parameter.append(sub_model_parameters[i][key])
        # weighted average
        delta = np.average(layer_parameter, axis=0, weights=w)
        base_parameter_dict[key] = (
            1-alpha) * base_parameter_dict[key] + alpha*delta


def fed_and_eval(base_index, w):
    base_env = make_vec_env(
        f"selected-bipedal-{subenv_dict[base_index]}-v0", n_envs=1, seed=seed)
    base_agent = ACKTR.load(f"./base_agent/{subenv_dict[base_index]}/model.zip")
    base_parameter_dict = base_agent.get_parameters()

    sub_model_parameters = []
    for subenv in subenv_dict.values():
        client_policy = ACKTR.load(
            f"./base{base_index}_client_model/{subenv}/policy.zip")
        sub_model_parameters.append(client_policy.get_parameters())

    aligned_agent = base_agent
    base_parameter_dict = aligned_agent.get_parameters()

    model_align(w, base_parameter_dict, sub_model_parameters, alpha=alpha)

    aligned_agent.load_parameters(base_parameter_dict)
    avg_reward, reward_std = evaluate_policy(
        aligned_agent, base_env, n_eval_episodes=100)

    print(f"base {base_index}, weight {w} done")
    return (avg_reward, reward_std)


if __name__ == "__main__":

    weights = grid_weights_gen(w_size=len(subenv_dict), grid_size=8)
    # weights = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]  # for test

    start_time = time.time()
    with mp.Pool(None) as p:
        test_results = p.starmap(fed_and_eval, product(range(4), weights))
    minutes, seconds = divmod(time.time() - start_time, 60)
    print(f"Time processed: {minutes:.0f}m {seconds:.0f}s")

    Path("log").mkdir(parents=True, exist_ok=True)
    for base_index in range(4):
        with open(f"log/wfedavg_log_base{base_index}.csv", "w", newline="") as f:
            wf = csv.writer(f)
            wf.writerow(weights)
            si = int(len(test_results)/4*base_index)
            ei = int(len(test_results)/4*(base_index+1))
            means, stds = [], []
            for mean_std in test_results[si:ei]:
                means.append(mean_std[0])
                stds.append(mean_std[1])
            wf.writerow(means)
            wf.writerow(stds)
