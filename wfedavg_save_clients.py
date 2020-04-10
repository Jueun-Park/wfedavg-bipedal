import gym
import gym_selected_bipedal
from stable_baselines import ACKTR
from stable_baselines.common import make_vec_env
from pathlib import Path
import argparse

from callback import save_rnd_dataset_callback
from info import subenv_dict, n_envs, seed, client_timesteps
parser = argparse.ArgumentParser()
parser.add_argument("--base-index", type=int)
args = parser.parse_args()
base_index = args.base_index

if __name__ == "__main__":
    base_agent = ACKTR.load(f"./model/{subenv_dict[base_index]}/model.zip")

    progress = ""
    for subenv in subenv_dict.values():
        env = make_vec_env(
            f"selected-bipedal-{subenv}-v0", n_envs=n_envs, seed=seed)
        learner = base_agent
        learner.env = env
        learner.verbose = 0
        
        learner.learn(total_timesteps=client_timesteps,
                      callback=save_rnd_dataset_callback)

        dir_name = f"base{base_index}_client_model/{subenv}"
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        learner.save(f"{dir_name}/policy.zip")
        del learner
        
        progress += " " + subenv
        print(f"progress: {progress}")
