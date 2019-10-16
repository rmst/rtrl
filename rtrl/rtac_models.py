from dataclasses import dataclass, InitVar

import gym
import torch
from rtrl.memory import collate, partition
from rtrl.nn import TanhNormalLayer
from torch.nn import Module, Linear, Sequential, ReLU, Conv2d, LeakyReLU
from torch.nn.functional import leaky_relu


@dataclass(eq=0)
class MlpRT(Module):
  observation_space: InitVar
  action_space: InitVar
  hidden_units: int = 256
  Linear: type = Linear

  def __post_init__(self, observation_space, action_space):
    super().__init__()
    assert isinstance(observation_space, gym.spaces.Tuple)
    input_dim = sum(s.shape[0] for s in observation_space)
    self.net = Sequential(
      self.Linear(input_dim, self.hidden_units), ReLU(),
      self.Linear(self.hidden_units, self.hidden_units), ReLU()
    )

    self.critic = Linear(self.hidden_units, 1)
    self.critic.weight.data.uniform_(-1e-3, 1e-3)
    self.critic.bias.data.uniform_(-1e-3, 1e-3)

    self.actor = TanhNormalLayer(self.hidden_units, action_space.shape[0])
    self.v_out = (self.critic,)

  def forward(self, x):
    assert isinstance(x, tuple)
    x = torch.cat(x, dim=1)
    h = self.net(x)
    v = self.critic(h)
    action_distribution = self.actor(h)
    return action_distribution, (v,)


@dataclass(eq=0)
class MlpRTDouble(torch.nn.Module):
  observation_space: InitVar
  action_space: InitVar
  hidden_units: int = 256
  Linear: type = Linear

  def __post_init__(self, observation_space, action_space):
    super().__init__()
    self.a = MlpRT(observation_space, action_space, hidden_units=self.hidden_units, Linear=self.Linear)
    self.b = MlpRT(observation_space, action_space, hidden_units=self.hidden_units, Linear=self.Linear)
    self.v_out = self.a.v_out + self.b.v_out

  def forward(self, x):
    action_distribution, (v0,) = self.a(x)
    _, (v1,) = self.b(x)
    return action_distribution, (v0, v1)

  def to(self, device):
    self.device = device
    return super().to(device=device)

  def act(self, obs, r, done, info, train=False):
    obs_col = collate((obs,), device=self.device)
    with torch.no_grad():
      action_distribution, _ = self.a(obs_col)
      action_col = action_distribution.sample() if train else action_distribution.sample_deterministic()
    action, = partition(action_col)
    return action, []


@dataclass(eq=0)
class ConvRTAC(Module):
  observation_space: InitVar
  action_space: InitVar
  hidden_units: int = 256

  num_critics: int = 1

  def __post_init__(self, observation_space, action_space):
    super().__init__()
    assert isinstance(observation_space, gym.spaces.Tuple)
    (img_sp, vec_sp), ac_sp = observation_space

    self.conv = big_conv(img_sp.shape[0])

    with torch.no_grad():
      conv_size = self.conv(torch.zeros((1, *img_sp.shape))).view(1, -1).size(1)

    self.lin1 = Linear(conv_size + vec_sp.shape[0] + ac_sp.shape[0], self.hidden_units)
    # self.lin2 = nn.Linear(hidden_units, a_space.shape[0])

    assert self.num_critics == 1
    self.critic = Linear(self.hidden_units, 1)
    self.actor = TanhNormalLayer(self.hidden_units, action_space.shape[0])
    self.v_out = (self.critic,)

  def forward(self, inp):
    (x, vec), action = inp
    x = x.type(torch.float32)
    x = x / 255 - 0.5
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    h = leaky_relu(self.lin1(torch.cat((x, vec, action), -1)))
    v = self.critic(h)
    action_distribution = self.actor(h)
    return action_distribution, (v,)


def big_conv(n):
  return Sequential(
    Conv2d(n, 32, 8, stride=2), LeakyReLU(),
    Conv2d(32, 32, 4, stride=2), LeakyReLU(),
    Conv2d(32, 32, 4, stride=2), LeakyReLU(),
    Conv2d(32, 32, 4, stride=1), LeakyReLU(),
  )