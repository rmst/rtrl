from dataclasses import InitVar, dataclass

import gym
import torch
from torch.nn.functional import leaky_relu

from rtrl.util import collate, partition
from torch.nn import Linear, Sequential, ReLU, ModuleList, Module
from rtrl.nn import TanhNormalLayer, SacLinear, big_conv


class ActorModule(Module):
  device = 'cpu'
  critic_output_layers = ()

  # noinspection PyMethodOverriding
  def to(self, device):
    self.device = device
    return super().to(device=device)

  def act(self, obs, r, done, info, train=False):
    obs_col = collate((obs,), device=self.device)
    with torch.no_grad():
      action_distribution = self.actor(obs_col)
      action_col = action_distribution.sample() if train else action_distribution.sample_deterministic()
    action, = partition(action_col)
    return action, []


class MlpActionValue(Sequential):
  def __init__(self, dim_obs, dim_action, hidden_units):
    super().__init__(
      SacLinear(dim_obs + dim_action, hidden_units), ReLU(),
      SacLinear(hidden_units, hidden_units), ReLU(),
      Linear(hidden_units, 1)
    )

  # noinspection PyMethodOverriding
  def forward(self, obs, action):
    x = torch.cat((*obs, action), 1)
    return super().forward(x)


class MlpPolicy(Sequential):
  def __init__(self, dim_obs, dim_action, hidden_units):
    super().__init__(
      SacLinear(dim_obs, hidden_units), ReLU(),
      SacLinear(hidden_units, hidden_units), ReLU(),
      TanhNormalLayer(hidden_units, dim_action)
    )

  def forward(self, obs):
    return super().forward(torch.cat(obs, 1))


class Mlp(ActorModule):
  def __init__(self, observation_space, action_space, hidden_units: int = 256, num_critics: int = 2):
    super().__init__()
    assert isinstance(observation_space, gym.spaces.Tuple)
    dim_obs = sum(space.shape[0] for space in observation_space)
    dim_action = action_space.shape[0]
    self.critics = ModuleList(MlpActionValue(dim_obs, dim_action, hidden_units) for _ in range(num_critics))
    self.actor = MlpPolicy(dim_obs, dim_action, hidden_units)
    self.critic_output_layers = [c[-1] for c in self.critics]


# === convolutional models =======================================================================================================
class ConvActor(Module):
  def __init__(self, observation_space, action_space, hidden_units: int = 256, Conv: type = big_conv):
    super().__init__()
    assert isinstance(observation_space, gym.spaces.Tuple)
    (img_sp, vec_sp), *aux = observation_space

    self.conv = Conv(img_sp.shape[0])
    with torch.no_grad():
      conv_size = self.conv(torch.zeros((1, *img_sp.shape))).view(1, -1).size(1)

    self.lin1 = torch.nn.Linear(conv_size + vec_sp.shape[0] + sum(sp.shape[0] for sp in aux), hidden_units)
    self.output_layer = TanhNormalLayer(hidden_units, action_space.shape[0])

  def forward(self, observation):
    (x, vec), *aux = observation
    x = x.type(torch.float32)
    x = x / 255 - 0.5
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    x = leaky_relu(self.lin1(torch.cat((x, vec, *aux), -1)))
    x = self.output_layer(x)
    return x


class ConvCritic(Module):
  def __init__(self, observation_space, action_space, hidden_units: int = 256, Conv: type = big_conv):
    super().__init__()
    assert isinstance(observation_space, gym.spaces.Tuple)
    (img_sp, vec_sp), *aux = observation_space

    self.conv = Conv(img_sp.shape[0])
    with torch.no_grad():
      conv_size = self.conv(torch.zeros((1, *img_sp.shape))).view(1, -1).size(1)

    self.net = Sequential(
      torch.nn.Linear(conv_size + vec_sp.shape[0] + sum(sp.shape[0] for sp in aux) + action_space.shape[0], hidden_units),
      torch.nn.LeakyReLU(),
      torch.nn.Linear(hidden_units, 1)
    )

  def __getitem__(self, item):
    return self.net[item]  # used for normalization in rtrl.sac:Agent

  def forward(self, observation, a):
    (x, vec), *aux = observation
    x = x.type(torch.float32)
    x = x / 255 - 0.5
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    x = torch.cat((x, vec, *aux, a), -1)
    x = self.net(x)
    return x


class ConvModel(ActorModule):
  def __init__(self, observation_space, action_space, num_critics: int = 2, hidden_units: int = 256):
    super().__init__()
    self.actor = ConvActor(observation_space, action_space, hidden_units)
    self.critics = ModuleList(ConvCritic(observation_space, action_space, hidden_units) for _ in range(num_critics))
    self.critic_output_layers = [c[-1] for c in self.critics]


# === Testing ==========================================================================================================
class TestMlp(ActorModule):
  def act(self, obs, r, done, info, train=False):
    return obs.copy(), {}
