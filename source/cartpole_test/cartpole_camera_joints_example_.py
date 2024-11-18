import gymnasium.spaces.box
import torch
import torch.nn as nn
import numpy as np

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.wrappers.torch.isaaclab_envs import IsaacLabWrapper
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from typing import Any, Mapping, Tuple, Union, Optional
from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper
import gymnasium
from skrl.envs.wrappers.torch.base import Wrapper


# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

class IsaacLabCustomWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Lab environment
        """
        super().__init__(env)

        self._reset_once = True
        self._observations = None
        self._info = {}

    @property
    def state_space(self) -> Union[gymnasium.Space, None]:
        """State space
        """
        try:
            return self._unwrapped.single_observation_space["critic"]
        except KeyError:
            pass
        try:
            return self._unwrapped.state_space
        except AttributeError:
            return None

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space
        """
        try:
            # return self._unwrapped.single_observation_space["policy"]
            return gymnasium.spaces.Box(low=-1.0,high=2.0,shape=(80*80*3+4,),dtype=np.float32)
        except:
            return self._unwrapped.observation_space["policy"]


    @property
    def action_space(self) -> gymnasium.Space:
        """Action space
        """
        try:
            return self._unwrapped.single_action_space
        except:
            return self._unwrapped.action_space

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        self._observations, reward, terminated, truncated, self._info = self._env.step(actions)
        observations = self._observation_to_tensor(self._observations["policy"])

        return observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        if self._reset_once:
            observations, self._info = self._env.reset()
            observations = self._observation_to_tensor(observations["policy"])
            self._reset_once = False

            return observations, self._info

        observations, self._info = self._env.reset()
        observations = self._observation_to_tensor(observations["policy"])

        return observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        return None

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()

    def _observation_to_tensor(self, observation: Any, space: Optional[gymnasium.Space] = None) -> torch.Tensor:
        """Convert the Gymnasium observation to a flat tensor

        :param observation: The Gymnasium observation to convert to a tensor
        :type observation: Any supported Gymnasium observation space

        :raises: ValueError if the observation space type is not supported

        :return: The observation as a flat tensor
        :rtype: torch.Tensor
        """
        observation_space = self._env.observation_space 
        space = space if space is not None else observation_space

        if isinstance(space, gymnasium.spaces.MultiDiscrete):
            return torch.tensor(observation, device=self.device, dtype=torch.int64).view(self.num_envs, -1)
        elif isinstance(observation, int):
            return torch.tensor(observation, device=self.device, dtype=torch.int64).view(self.num_envs, -1)
        elif isinstance(observation, np.ndarray):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        elif isinstance(space, gymnasium.spaces.Discrete):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        elif isinstance(space, gymnasium.spaces.Box):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        elif isinstance(space, gymnasium.spaces.Dict):
            tmp = torch.cat([self._observation_to_tensor(observation[k], space[k]) \
                for k in sorted(space.keys())], dim=-1).view(self.num_envs, -1)
            return tmp
        else:
            raise ValueError(f"Observation space type {type(space)} not supported. Please report this issue")
        

class GymnasiumCustomWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Gymnasium environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Gymnasium environment
        """
        super().__init__(env)

        self._vectorized = False
        try:
            if isinstance(env, gymnasium.vector.VectorEnv) or isinstance(env, gymnasium.experimental.vector.VectorEnv):
                self._vectorized = True
                self._reset_once = True
                self._observation = None
                self._info = None
        except Exception as e:
            logger.warning(f"Failed to check for a vectorized environment: {e}")

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space
        """
        if self._vectorized:
            return self._env.single_observation_space
        return self._env.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space
        """
        if self._vectorized:
            return self._env.single_action_space
        return self._env.action_space

    def _observation_to_tensor(self, observation: Any, space: Optional[gymnasium.Space] = None) -> torch.Tensor:
        """Convert the Gymnasium observation to a flat tensor

        :param observation: The Gymnasium observation to convert to a tensor
        :type observation: Any supported Gymnasium observation space

        :raises: ValueError if the observation space type is not supported

        :return: The observation as a flat tensor
        :rtype: torch.Tensor
        """
        observation_space = self._env.observation_space if self._vectorized else self.observation_space
        space = space if space is not None else observation_space

        if self._vectorized and isinstance(space, gymnasium.spaces.MultiDiscrete):
            return torch.tensor(observation, device=self.device, dtype=torch.int64).view(self.num_envs, -1)
        elif isinstance(observation, int):
            return torch.tensor(observation, device=self.device, dtype=torch.int64).view(self.num_envs, -1)
        elif isinstance(observation, np.ndarray):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        elif isinstance(space, gymnasium.spaces.Discrete):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        elif isinstance(space, gymnasium.spaces.Box):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        elif isinstance(space, gymnasium.spaces.Dict):
            tmp = torch.cat([self._observation_to_tensor(observation[k], space[k]) \
                for k in sorted(space.keys())], dim=-1).view(self.num_envs, -1)
            return tmp
        else:
            raise ValueError(f"Observation space type {type(space)} not supported. Please report this issue")

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        observation, reward, terminated, truncated, info = self._env.step(actions)

        # convert response to torch
        observation = self._observation_to_tensor(observation)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
        truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)

        # save observation and info for vectorized envs
        if self._vectorized:
            self._observation = observation
            self._info = info

        return observation, reward, terminated, truncated, info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        # handle vectorized environments (vector environments are autoreset)
        if self._vectorized:
            if self._reset_once:
                observation, self._info = self._env.reset()
                self._observation = self._observation_to_tensor(observation)
                self._reset_once = False
            return self._observation, self._info

        observation, info = self._env.reset()
        return self._observation_to_tensor(observation), info

    def render(self, *args, **kwargs) -> Any:
        """Render the environment
        """
        if self._vectorized:
            return self._env.call("render", *args, **kwargs)
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()


# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.features_extractor = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                                nn.ReLU(),
                                                nn.Flatten())

        self.net = nn.Sequential(nn.Linear(2308, 512),
                                 nn.ELU())

        self.mean_layer = nn.Linear(512, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))

        self.value_layer = nn.Linear(512, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)


    def compute(self, inputs, role):
        states = inputs["states"]
        space = self.tensor_to_space(states, gymnasium.spaces.Dict({"camera": gymnasium.spaces.Box(low=-1.0,high=2.0,shape=(1,80,80, 3),dtype=np.float32),
                                                                    "joints": gymnasium.spaces.Box(low=-np.inf,high=np.inf,shape=(1,4),dtype=np.float32)}))

        if role == "policy":
            features = self.features_extractor(space["camera"][:,0].permute(0, 3, 1, 2))
            self._shared_output = self.net(torch.cat([features, space["joints"].view(states.shape[0],-1)],dim=-1))
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            features = self.features_extractor(space["camera"][:,0].permute(0, 3, 1, 2))
            self._shared_output = self.net(torch.cat([features, space["joints"].view(states.shape[0],-1)],dim=-1))
            return self.value_layer(self._shared_output), {}
    




# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Cartpole-RGB-Camera-Direct-v0")
env = IsaacLabCustomWrapper(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=64, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"] # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 64  # memory_size
cfg["learning_epochs"] = 4
cfg["mini_batches"] = 1  # 16 * 512 / 8192
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 2.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = True
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 16
cfg["experiment"]["checkpoint_interval"] = 80
cfg["experiment"]["directory"] = "runs/torch/Isaac-Cartpole-RGB-Camera-v0"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 6000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()


# # ---------------------------------------------------------
# # comment the code above: `trainer.train()`, and...
# # uncomment the following lines to evaluate a trained agent
# # ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# # download the trained agent's checkpoint from Hugging Face Hub and load it
# path = download_model_from_huggingface("skrl/IsaacOrbit-Isaac-Cartpole-v0-PPO", filename="agent.pt")
# agent.load(path)

# # start evaluation
# trainer.eval()
