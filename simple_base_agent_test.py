import gym
import gym_selected_bipedal
from stable_baselines import ACKTR
from stable_baselines.common import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
import csv
from pathlib import Path
import multiprocessing as mp
import time

from info import subenv_dict, seed


def eval_base_agent(agent_index):
    Path("log").mkdir(parents=True, exist_ok=True)
    file = open(
        f"log/agent{agent_index}_simple_agent_test.csv", "w", newline="")
    writer = csv.writer(file)
    mean_result = []
    std_result = []
    agent = ACKTR.load(f"./base_agent/{subenv_dict[agent_index]}/model.zip")
    for env_index in range(4):
        env = gym.make(f"selected-bipedal-{subenv_dict[env_index]}-v0")
        env.seed = seed
        mean, std = evaluate_policy(agent, env, n_eval_episodes=100)
        mean_result.append(mean)
        std_result.append(std)
    writer.writerow(mean_result)
    writer.writerow(std_result)
    file.close()
    print(f">>> Agent {agent_index}:")
    print(mean_result)
    print(std_result)
    return


if __name__ == "__main__":
    # processes = []
    # for agent_index in range(4):
    #     p = mp.Process(target=eval_base_agent, args=(agent_index,))
    #     processes.append(p)
    #     p.start()
    # for p in processes:
    #     p.join()
    start_time = time.time()
    with mp.Pool(None) as p:
        p.map(eval_base_agent, [i for i in range(4)])
    minutes, seconds = divmod(time.time() - start_time, 60)
    print(f"Time processed: {minutes:.0f}m {seconds:.0f}s")
