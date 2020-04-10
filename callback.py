import numpy as np
import pickle

from info import n_envs


n_steps_for_callback = 1000  # batch size 20k
global rnd_training_dataset
rnd_training_dataset = []


def save_rnd_dataset_callback(_locals, _globals):
    global rnd_training_dataset
    agent = _locals["self"]
    num_timesteps = agent.num_timesteps
    obs = _locals["obs"]

    # make batch
    batch_len = len(obs) // n_envs
    np.random.shuffle(obs)
    obs = obs[:batch_len]
    rnd_training_dataset += list(obs)

    if (num_timesteps // batch_len) % n_steps_for_callback == 0:
        print(
            f"> data gathering for RND. data size: {len(rnd_training_dataset)}")
    if _locals["update"] == _locals["total_timesteps"] // agent.n_batch:
        print(
            f"> save RND training dataset. data size: {len(rnd_training_dataset)}")
        env_name = agent.env.envs[0].__str__()
        env_name = env_name.split("<")[4]
        env_name = env_name.split("-")[2]
        with open(f"rnd_dataset/{env_name}.pkl", "wb") as f:
            pickle.dump(rnd_training_dataset, f)
        rnd_training_dataset = []
    return True
