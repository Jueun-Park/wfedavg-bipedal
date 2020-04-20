from stable_baselines.common.callbacks import BaseCallback
import numpy as np
import pickle
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

from info import n_envs


# stable-baselines==2.10.0 (+user fix)
class SaveRNDDatasetCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, base_index, subenv_seed=None, verbose=0):
        super(SaveRNDDatasetCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.base_index = base_index
        self.rnd_training_dataset = []
        self.seed = subenv_seed

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        obs = self.locals["obs"]
        batch_len = len(obs) // n_envs
        np.random.shuffle(obs)
        obs = obs[:batch_len]
        self.rnd_training_dataset += list(obs)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        env_name = self.model.env.envs[0].__str__()
        env_name = env_name.split("<")[4]
        env_name = env_name.split("-")[2]
        dir_name = f"rnd_dataset/base{self.base_index}"
        if self.seed is not None:
            dir_name += f"_{self.seed}"
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        with open(f"{dir_name}/{env_name}.pkl", "wb") as f:
            pickle.dump(self.rnd_training_dataset, f)
        print(
            f"> save RND training dataset. Path: {dir_name}/{env_name}.pkl / data size: {len(rnd_training_dataset)}")


# ======
# stable-baselines==2.9.0
base_index = 3

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
        dir_name = f"rnd_dataset/base{base_index}"
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        with open(f"{dir_name}/{env_name}.pkl", "wb") as f:
            pickle.dump(rnd_training_dataset, f)
        rnd_training_dataset = []
    return True
