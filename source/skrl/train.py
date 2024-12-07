import gymnasium.spaces.box
import torch
import torch.nn as nn
import numpy as np

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO_RNN as PPO, PPO_DEFAULT_CONFIG
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
            return self._unwrapped.single_observation_space["policy"]
            # return gymnasium.spaces.Box(low=-1.0,high=2.0,shape=(5,),dtype=np.float32)
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
        # observations = self._observation_to_tensor(self._observations["policy"])

        return self.observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        if self._reset_once:
            observations, self._info = self._env.reset()
            # observations = self._observation_to_tensor(observations["policy"])
            self._reset_once = False

            return observations, self._info

        observations, self._info = self._env.reset()
        # observations = self._observation_to_tensor(observations["policy"])

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

    

# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=1, num_layers=5, hidden_size=512, sequence_length=128):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        states = inputs["states"]
        # space = self.tensor_to_space(states, gymnasium.spaces.Dict({"position": gymnasium.spaces.Box(low=-np.inf,high=np.inf,shape=(1,5),dtype=np.float32)}))
        # states = space["position"]

        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # training
        if self.training:
            print("STATES: "+str(states[1]))
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        # Desired action_space is -0.01 to 0.01
        return self.net(rnn_output), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, num_layers=5, hidden_size=512, sequence_length=128):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        states = inputs["states"]
        # space = self.tensor_to_space(states, gymnasium.spaces.Dict({"position": gymnasium.spaces.Box(low=-np.inf,high=np.inf,shape=(1,5),dtype=np.float32)}))
        # states = space["position"]

        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # training
        if self.training:
            
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length

            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.net(rnn_output), {"rnn": [rnn_states[0], rnn_states[1]]}



# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Franka-Pouring-Direct-v0")
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=1, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True, num_envs=env.num_envs)
models["value"] = Value(env.observation_space, env.action_space, device, num_envs=env.num_envs)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1  # memory_size
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
cfg["entropy_loss_scale"] = 0
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = True
cfg["state_preprocessor"] = None
cfg["state_preprocessor_kwargs"] = {}
cfg["value_preprocessor"] = None
cfg["value_preprocessor_kwargs"] = {}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 16
cfg["experiment"]["checkpoint_interval"] = 80
cfg["experiment"]["directory"] = "runs/torch/pouring_ppo"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 32000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# # start training
trainer.train()


# # ---------------------------------------------------------
# # comment the code above: `trainer.train()`, and...
# # uncomment the following lines to evaluate a trained agent
# # ---------------------------------------------------------

# # load model
# import os

# PATH = os.path.dirname(os.path.realpath(__file__))
# path = f"/{PATH}/../../runs/torch/pouring_ppo/Basic/checkpoints/best_agent.pt"
# agent.load(path)

# # start evaluation
# trainer.eval()
