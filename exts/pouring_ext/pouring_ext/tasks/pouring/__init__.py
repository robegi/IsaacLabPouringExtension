"""
Liquid manipulation task, Direct implementation
"""


import gymnasium as gym

from . import agents
from .franka_pouring_env import FrankaPouringEnv, FrankaPouringEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Franka-Pouring-Direct-v0",
    entry_point="pouring_ext.tasks.pouring:FrankaPouringEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaPouringEnvCfg,      
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPouringPPORunnerCfg",
    },
)

