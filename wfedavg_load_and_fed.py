import gym
import gym_selected_bipedal
from stable_baselines import ACKTR
from stable_baselines.common import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
import numpy as np
import argparse
import csv
from pathlib import Path

from modules.gen_weights import grid_weights_gen
from info import subenv_dict, seed, alpha

parser = argparse.ArgumentParser()
parser.add_argument("--base-index", type=int)
args = parser.parse_args()
base_index = args.base_index


def model_align(w, base_parameter_dict, sub_model_parameters, alpha=0.5):
    keys = base_parameter_dict.keys()
    for key in keys:
        layer_parameter = []
        for i in range(4):
            layer_parameter.append(sub_model_parameters[i][key])
        # weighted average
        delta = np.average(layer_parameter, axis=0, weights=w)
        base_parameter_dict[key] = (
            1-alpha) * base_parameter_dict[key] + alpha*delta


if __name__ == "__main__":
    base_env = make_vec_env(
        f"selected-bipedal-{subenv_dict[base_index]}-v0", n_envs=1, seed=seed)
    base_agent = ACKTR.load(f"./model/{subenv_dict[base_index]}/model.zip")
    base_parameter_dict = base_agent.get_parameters()

    sub_model_parameters = []
    for subenv in subenv_dict.values():
        client_policy = ACKTR.load(
            f"./base{base_index}_client_model/{subenv}/policy.zip")
        sub_model_parameters.append(client_policy.get_parameters())

    weights = grid_weights_gen(w_size=len(subenv_dict), grid_size=8)
    # weights = [[1,0,0,0], [0,1,0,0],[0,0,1,0],[0,0,0,1]]  # for test
    w_labels = []
    test_rewards = []
    for i, w in enumerate(weights):
        aligned_agent = base_agent
        base_parameter_dict = aligned_agent.get_parameters()
        model_align(w, base_parameter_dict, sub_model_parameters, alpha=alpha)
        aligned_agent.load_parameters(base_parameter_dict)
        avg_reward, reward_std = evaluate_policy(aligned_agent, base_env, n_eval_episodes=100)
        w_labels.append(w)
        test_rewards.append(avg_reward)
        if i % 10 == 0:
            print(f"w test in progressing {i}/{len(weights)}")

    Path("log").mkdir(parents=True, exist_ok=True)
    with open(f"log/wfedavg_log_base{base_index}.csv", "w", newline="") as f:
        wf = csv.writer(f)
        wf.writerow(w_labels)
        wf.writerow(test_rewards)
