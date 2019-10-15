from dataclasses import InitVar, dataclass

import gym
import torch
from rtrl.memory import collate, partition
from torch.nn import Linear, Sequential, ReLU, ModuleList, Module, LeakyReLU, Conv2d
from rtrl.nn import TanhNormalLayer
from rtrl.util import cached_property
from torch.nn.functional import leaky_relu
import numpy as np


class RlkitLinear(Linear):
  def __init__(self, *args):
    super().__init__(*args)
    fan_in = self.weight.shape[0]
    bound = 1. / np.sqrt(fan_in)
    self.weight.data.uniform_(-bound, bound)
    self.bias.data.fill_(0.1)


class MlpActionValue(Sequential):
  def __init__(self, dim_obs, dim_action, hidden_units):
    super().__init__(
      Linear(dim_obs + dim_action, hidden_units), ReLU(),
      Linear(hidden_units, hidden_units), ReLU(),
      Linear(hidden_units, 1)
    )

  # noinspection PyMethodOverriding
  def forward(self, obs, action):
    x = torch.cat((*obs, action), 1)
    return super().forward(x)


class MlpValue(Sequential):
  def __init__(self, dim_obs, dim_action, hidden_units):
    super().__init__(
      Linear(dim_obs, hidden_units), ReLU(),
      Linear(hidden_units, hidden_units), ReLU(),
      Linear(hidden_units, 1)
    )

  def forward(self, obs):
    return super().forward(torch.cat(obs, -1))


class MlpPolicy(Sequential):
  def __init__(self, dim_obs, dim_action, hidden_units):
    super().__init__(
      Linear(dim_obs, hidden_units), ReLU(),
      Linear(hidden_units, hidden_units), ReLU(),
      TanhNormalLayer(hidden_units, dim_action)
    )

  def forward(self, obs):
    return super().forward(torch.cat(obs, 1))


@dataclass(eq=0)
class Mlp(Module):
  observation_space: InitVar
  action_space: InitVar

  hidden_units: int = 256
  num_critics: int = 2

  device = 'cpu'

  def __post_init__(self, observation_space, action_space):
    super().__init__()
    assert isinstance(observation_space, gym.spaces.Tuple)
    dim_obs = sum(space.shape[0] for space in observation_space)
    dim_action = action_space.shape[0]
    self.critics = ModuleList(MlpActionValue(dim_obs, dim_action, self.hidden_units) for _ in range(self.num_critics))
    self.value = MlpValue(dim_obs, dim_action, self.hidden_units)
    self.actor = MlpPolicy(dim_obs, dim_action, self.hidden_units)

  def to(self, device):
    self.device = device
    return super().to(device=device)

  def act(self, obs, r, done, info):
    obs_col = collate((obs,), device=self.device)
    with torch.no_grad():
      action_distribution = self.actor(obs_col)
      action_col = action_distribution.sample()
    action, = partition(action_col)
    return action, []


# === Testing ==========================================================================================================
class TestMlp(Mlp):
  def act(self, obs: torch.tensor, r, done, info):
    return obs.copy(), {}

# ======================================================================================================================


@dataclass(eq=0)
class MlpRT(Module):
  observation_space: InitVar
  action_space: InitVar
  hidden_units: int = 256
  # num_critics: int = 1

  def __post_init__(self, observation_space, action_space):
    super().__init__()
    assert isinstance(observation_space, gym.spaces.Tuple)
    input_dim = sum(s.shape[0] for s in observation_space)
    self.net = Sequential(
      RlkitLinear(input_dim, self.hidden_units), ReLU(),
      RlkitLinear(self.hidden_units, self.hidden_units), ReLU()
    )

    self.critic = Linear(self.hidden_units, 1)
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

  def __post_init__(self, observation_space, action_space):
    super().__init__()
    self.a = MlpRT(observation_space, action_space, hidden_units=self.hidden_units)
    self.b = MlpRT(observation_space, action_space, hidden_units=self.hidden_units)
    self.v_out = self.a.v_out + self.b.v_out

  def forward(self, x):
    action_distribution, (v0,) = self.a(x)
    _, (v1,) = self.b(x)
    return action_distribution, (v0, v1)

  def to(self, device):
    self.device = device
    return super().to(device=device)

  def act(self, obs, r, done, info):
    obs_col = collate((obs,), device=self.device)
    with torch.no_grad():
      action_distribution, _ = self.a(obs_col)
      action_col = action_distribution.sample()
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
#
#
# class SeparateRT(nn.Module):
#   class HL(Mlp.HL):
#     class L(RlkitHiddenLinear): pass
#
#   class LCL(agents.nn.Linear):
#     pass
#
#   class LPL(TanhNormalLayer): pass
#
#   hidden_units: int = 256
#   num_critics: int = 1
#
#   def __init__(self, ob_space, a_space, **kwargs):
#     super().__init__()
#
#     apply_kwargs(self, kwargs)
#     s = STATE({k: v.shape for k, v in ob_space.spaces.items()})
#     self.dim_obs = s.s[0] + (s.a[0] if s.a is not None else 0)
#     self.a_space = a_space
#
#     self.critic = nn.Sequential(self.HL(self.dim_obs, self.hidden_units),
#                                 self.HL(self.hidden_units, self.hidden_units),
#                                 self.LCL(self.hidden_units, self.num_critics))
#
#     self.actor = nn.Sequential(self.HL(self.dim_obs, self.hidden_units),
#                                self.HL(self.hidden_units, self.hidden_units),
#                                self.LPL(self.hidden_units, self.a_space.shape[0]))
#     self.v_out = (self.critic[-1],)
#
#   def forward(self, x):
#     v = self.critic(x.s)
#     a = self.actor(x.s)
#     return (a, None), tuple(v[:, i:i + 1] for i in range(self.num_critics))
#
#

#
#
#
# # CONV MODELS
#
# def dqn_conv(n):
#   return nn.Sequential(
#       nn.Conv2d(n, 32, kernel_size=8, stride=4),
#       nn.ReLU(),
#       nn.Conv2d(32, 64, kernel_size=4, stride=2),
#       nn.ReLU(),
#       nn.Conv2d(64, 64, kernel_size=3, stride=1),
#       nn.ReLU()
#     )
#






#
#
# class ConvActor(nn.Module):
#   def __init__(self, ob_space, a_space, hidden_units):
#     super().__init__()
#     s = STATE({k: v.shape for k, v in ob_space.spaces.items()})
#     self.conv = big_conv(s.vis[0])
#
#     with torch.no_grad():
#       conv_size = self.conv(torch.zeros((1, *s.vis))).view(1, -1).size(1)
#
#     self.lin1 = nn.Linear(conv_size + s.s[0], hidden_units)
#     # self.lin2 = nn.Linear(hidden_units, a_space.shape[0])
#     self.lpl = TanhNormalLayer(hidden_units, a_space.shape[0])
#
#   def forward(self, inp: STATE):
#     x = inp.vis.type(torch.float32)
#     x = x / 255 - 0.5
#     x = self.conv(x)
#     x = x.view(x.size(0), -1)
#     x = fu.leaky_relu(self.lin1(torch.cat((x, inp.s), -1)))
#     x = self.lpl(x)
#     return x
#
#
# class ConvCritic(nn.Module):
#   def __init__(self, ob_space, a_space, hidden_units):
#     super().__init__()
#     s = STATE({k: v.shape for k, v in ob_space.spaces.items()})
#     self.conv = big_conv(s.vis[0])
#
#     with torch.no_grad():
#       conv_size = self.conv(torch.zeros((1, *s.vis))).view(1, -1).size(1)
#
#     self.lin1 = nn.Linear(conv_size + s.s[0] + a_space.shape[0], hidden_units)
#     self.lin2 = nn.Linear(hidden_units, 1)
#     self.other = [self.lin2]
#
#   def forward(self, inp: STATE, a):
#     x = inp.vis.type(torch.float32)
#     x = x / 255 - 0.5
#     x = self.conv(x)
#     x = x.view(x.size(0), -1)
#     x = fu.leaky_relu(self.lin1(torch.cat((x, inp.s, a.a), -1)))
#     x = self.lin2(x)
#     return x
#
#
# class ConvValue(nn.Module):
#   def __init__(self, ob_space, a_space, hidden_units):
#     super().__init__()
#     s = STATE({k: v.shape for k, v in ob_space.spaces.items()})
#     self.conv = big_conv(s.vis[0])
#
#     with torch.no_grad():
#       conv_size = self.conv(torch.zeros((1, *s.vis))).view(1, -1).size(1)
#
#     self.lin1 = nn.Linear(conv_size + s.s[0], hidden_units)
#     self.lin2 = nn.Linear(hidden_units, 1)
#     self.other = [self.lin2]
#
#   def forward(self, inp: STATE):
#     x = inp.vis.type(torch.float32)
#     x = x / 255 - 0.5
#     x = self.conv(x)
#     x = x.view(x.size(0), -1)
#     x = fu.leaky_relu(self.lin1(torch.cat((x, inp.s), -1)))
#     x = self.lin2(x)
#     return x
#
#
# class ConvModel(nn.Module):
#   num_critics: int = 2
#   hidden_units: int = 256
#
#   def __init__(self, ob_space, a_space, **kwargs):
#     super().__init__()
#     apply_kwargs(self, kwargs)
#     self.actor = ConvActor(ob_space, a_space, self.hidden_units)
#     self.critics = nn.ModuleList(ConvCritic(ob_space, a_space, self.hidden_units) for _ in range(self.num_critics))
#     self.value = ConvValue(ob_space, a_space, self.hidden_units)
#
