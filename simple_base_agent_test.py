import gym
import gym_selected_bipedal
from stable_baselines import ACKTR
from stable_baselines.common import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
import csv

from info import subenv_dict


if __name__ == "__main__":
    file = open("log/simple_agent_test.csv", "w", newline="")
    writer = csv.writer(file)
    mean_result = [[] for _ in range(4)]
    std_result = [[] for _ in range(4)]
    for agent_index in range(4):
        agent = ACKTR.load(f"./model/{subenv_dict[agent_index]}/model.zip")
        for env_index in range(4):
            env = make_vec_env(
                f"selected-bipedal-{subenv_dict[env_index]}-v0", n_envs=1, seed=0)
            mean, std = evaluate_policy(agent, env, n_eval_episodes=100)
            mean_result[agent_index].append(mean)
            std_result[agent_index].append(std)
    for i in range(len(mean_result)):
        writer.writerow(mean_result[i])
    for i in range(len(std_result)):
        writer.writerow(std_result[i])    
    file.close()
