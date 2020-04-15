import gym
import gym_selected_bipedal
from stable_baselines import ACKTR
from stable_baselines.common import make_vec_env
from pathlib import Path
import multiprocessing as mp
from itertools import product
import time

from callback import save_rnd_dataset_callback
from info import subenv_dict, n_envs, seed, client_timesteps


def save_client(args):
    base_index, subenv_id = args
    base_agent = ACKTR.load(f"./base_agent/{subenv_dict[base_index]}/model.zip")

    subenv = subenv_dict[subenv_id]
    env = make_vec_env(
        f"selected-bipedal-{subenv}-v0", n_envs=n_envs, seed=seed)
    learner = base_agent
    learner.env = env
    learner.verbose = 0

    learner.learn(total_timesteps=client_timesteps,
                #   callback=save_rnd_dataset_callback,  # TODO: fix rnd callback
                  )

    dir_name = f"base{base_index}_client_model/{subenv}"
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    learner.save(f"{dir_name}/policy.zip")
    print(f"base {base_index} sub-env {subenv} done")


if __name__ == "__main__":
    start_time = time.time()
    args = []
    for arg in product(range(4), range(4)):
        args.append(arg)
    print(args)

    with mp.Pool(None) as p:
        p.map(save_client, args)
    
    minutes, seconds = divmod(time.time() - start_time, 60)
    print(f"Time processed: {minutes:.0f}m {seconds:.0f}s")
