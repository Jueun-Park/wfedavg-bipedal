import pickle
import gym
import gym_selected_bipedal
import numpy as np
import multiprocessing as mp
from itertools import product
import time

from modules.rnd import RandomNetworkDistillation
from info import subenv_dict, seed


def train_rnd(base_id, subenv_seed=None):
    test_env_id = subenv_dict[base_id]
    subenv_id = test_env_id
    with open(f"rnd_dataset/base{base_id}_{subenv_seed}/{subenv_id}.pkl", "rb") as f:
        rnd_training_dataset = pickle.load(f)
    print(
        f"> rnd training data size: {len(rnd_training_dataset)}")
    env = gym.make(f"selected-bipedal-{test_env_id}-v0")
    rnd = RandomNetworkDistillation(
        env.observation_space.shape[0], use_cuda=True, tensorboard=False, verbose=0)
    rnd.learn(np.array(rnd_training_dataset), n_steps=2000)
    save_path = f"ho_base{base_id}_client_model/{subenv_seed}/rnd"
    rnd.save(path=save_path)
    print(f"save {save_path}")


if __name__ == "__main__":
    start_time = time.time()
    with mp.Pool(None) as pool:
        pool.starmap(
            train_rnd,
            product(
                subenv_dict.keys(),
                [seed + i for i in range(4)]),
            chunksize=4,
            )
    minutes, seconds = divmod(time.time() - start_time, 60)
    print(f"Time processed: {minutes:.0f}m {seconds:.0f}s")
