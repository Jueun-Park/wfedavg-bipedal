import pickle
import gym
import gym_selected_bipedal
import numpy as np

from modules.rnd import RandomNetworkDistillation
from info import subenv_dict


if __name__ == "__main__":
    for subenv_id in subenv_dict.values():
        with open(f"rnd_dataset/{subenv_id}.pkl", "rb") as f:
            rnd_training_dataset = pickle.load(f)
        print(
            f"> rnd training data size: {len(rnd_training_dataset)}")
        for base_id, test_env_id in subenv_dict.items():
            env = gym.make(f"selected-bipedal-{test_env_id}-v0")
            rnd = RandomNetworkDistillation(
                env.observation_space.shape[0], use_cuda=False, tensorboard=False)
            rnd.learn(np.array(rnd_training_dataset), n_steps=2000)
            rnd.save(path=f"base{base_id}_client_model/{subenv_id}/rnd")
