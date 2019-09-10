from dataclasses import InitVar, dataclass
from functools import partial

import torch
from rtrl.memory import collate, partition
from torch import nn
from torch.distributions import Normal, Distribution
from torch.nn import functional as fu, Linear, Sequential, ReLU, ModuleList, Module


# noinspection PyAbstractClass
class TanhNormal(Distribution):
  """ Represent distribution of X where
      X ~ tanh(Z)
      Z ~ N(mean, std)
  """
  def __init__(self, normal_mean, normal_std, epsilon=1e-6):
    """
    :param normal_mean: Mean of the normal distribution
    :param normal_std: Std of the normal distribution
    :param epsilon: Numerical stability epsilon when computing log-prob.
    """
    self.normal_mean = normal_mean
    self.normal_std = normal_std
    self.normal = Normal(normal_mean, normal_std)
    self.epsilon = epsilon
    super().__init__(self.normal.batch_shape, self.normal.event_shape)

  def log_prob(self, x):
    assert isinstance(x, dict)
    assert x["value"].dim() == 2 and x["pre_tanh_value"].dim() == 2
    return self.normal.log_prob(x["pre_tanh_value"]) - torch.log(
      1 - x["value"] * x["value"] + self.epsilon
    )

  def sample(self, sample_shape=torch.Size()):
    z = self.normal.sample(sample_shape)
    out = torch.tanh(z)
    return dict(value=out, pre_tanh_value=z)

  def rsample(self, sample_shape=torch.Size()):
    z = self.normal.rsample(sample_shape)
    out = torch.tanh(z)
    return dict(value=out, pre_tanh_value=z)


# noinspection PyAbstractClass
class Independent(torch.distributions.Independent):
  def sample_deterministic(self):
    return torch.tanh(self.base_dist.normal_mean)


class TanhNormalLayer(nn.Module):
  def __init__(self, n, m):
    super().__init__()

    self.lin_mean = torch.nn.Linear(n, m)
    # self.lin_mean.weight.data
    # self.lin_mean.bias.data

    self.lin_std = torch.nn.Linear(n, m)
    self.lin_std.weight.data.uniform_(-1e-3, 1e-3)
    self.lin_std.bias.data.uniform_(-1e-3, 1e-3)

  def forward(self, x):
    mean = self.lin_mean(x)
    log_std = self.lin_std(x)
    log_std = torch.clamp(log_std, 2, -20)
    std = torch.exp(log_std)
    # a = TanhTransformedDist(Independent(Normal(m, std), 1))
    a = Independent(TanhNormal(mean, std), 1)
    return a


class MlpActionValue(Sequential):
  def __init__(self, dim_obs, dim_action, hidden_units):
    super().__init__(
      Linear(dim_obs + dim_action, hidden_units), ReLU(),
      Linear(hidden_units, hidden_units), ReLU(),
      Linear(hidden_units, 1)
    )

  # noinspection PyMethodOverriding
  def forward(self, obs, action):
    x = torch.cat((obs['vector'], action['value']), 1)
    return super().forward(x)


class MlpValue(Sequential):
  def __init__(self, dim_obs, dim_action, hidden_units):
    super().__init__(
      Linear(dim_obs, hidden_units), ReLU(),
      Linear(hidden_units, hidden_units), ReLU(),
      Linear(hidden_units, 1)
    )

  def forward(self, obs):
    return super().forward(obs['vector'])


class MlpPolicy(Sequential):
  def __init__(self, dim_obs, dim_action, hidden_units):
    super().__init__(
      Linear(dim_obs, hidden_units), ReLU(),
      Linear(hidden_units, hidden_units), ReLU(),
      TanhNormalLayer(hidden_units, dim_action)
    )

  def forward(self, obs):
    return super().forward(obs['vector'])


@dataclass(unsafe_hash=True)
class Mlp(Module):
  observation_space: InitVar
  action_space: InitVar

  hidden_units: int = 256
  num_critics: int = 2

  def __post_init__(self, observation_space, action_space):
    super().__init__()
    dim_obs = observation_space.spaces['vector'].shape[0]
    dim_action = action_space.spaces['value'].shape[0]
    self.critics = ModuleList(MlpActionValue(dim_obs, dim_action, self.hidden_units) for _ in range(self.num_critics))
    self.value = MlpValue(dim_obs, dim_action, self.hidden_units)
    self.actor = MlpPolicy(dim_obs, dim_action, self.hidden_units)

  def act(self, obs, r, done, info):
    obs_col = collate((obs,))
    with torch.no_grad():
      action_distribution = self.actor(obs_col)
      action_col = action_distribution.sample()
    action, = partition(action_col)
    return action, {}


# class MlpRT(nn.Module):
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
#     self.net = nn.Sequential(self.HL(self.dim_obs, self.hidden_units),
#                              self.HL(self.hidden_units, self.hidden_units))
#
#     self.critic = self.LCL(self.hidden_units, self.num_critics)
#     self.actor = self.LPL(self.hidden_units, self.a_space.shape[0])
#     self.v_out = (self.critic,)
#
#   def forward(self, x):
#     h = self.net(x.s)
#     v = self.critic(h)
#     a = self.actor(h)
#     return (a, None), tuple(v[:, i:i + 1] for i in range(self.num_critics))
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
# class MlpRTDouble(torch.nn.Module):
#   class M(MlpRT): pass
#
#   hidden_units: int = 256
#
#   def __init__(self, ob_space, a_space, **kwargs):
#     super().__init__()
#     apply_kwargs(self, kwargs)
#     self.a = self.M(ob_space, a_space, hidden_units=self.hidden_units)
#     self.b = self.M(ob_space, a_space, hidden_units=self.hidden_units)
#     self.v_out = self.a.v_out + self.b.v_out
#
#   def forward(self, x):
#     a0, v0 = self.a(x)
#     a1, v1 = self.b(x)
#     return (a0[0], a1[0]), (v0[0], v1[0])
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
# def big_conv(n):
#   return nn.Sequential(
#     nn.Conv2d(n, 32, 8, stride=2), nn.LeakyReLU(),
#     nn.Conv2d(32, 32, 4, stride=2), nn.LeakyReLU(),
#     nn.Conv2d(32, 32, 4, stride=2), nn.LeakyReLU(),
#     nn.Conv2d(32, 32, 4, stride=1), nn.LeakyReLU(),
#   )
#
# class ConvRTAC(nn.Module):
#   class LCL(agents.nn.Linear):
#     pass
#
#   class LPL(TanhNormalLayer): pass
#
#   num_critics = 1
#
#   def __init__(self, ob_space, a_space, hidden_units):
#     super().__init__()
#     s = STATE({k: v.shape for k, v in ob_space.spaces.items()})
#     self.a_space = a_space
#
#     self.conv = big_conv(s.vis[0])
#     with torch.no_grad():
#       conv_size = self.conv(torch.zeros((1, *s.vis))).view(1, -1).size(1)
#
#     self.lin1 = nn.Linear(conv_size + s.s[0], hidden_units)
#     # self.lin2 = nn.Linear(hidden_units, a_space.shape[0])
#
#     self.critic = self.LCL(hidden_units, self.num_critics)
#     self.actor = self.LPL(hidden_units, self.a_space.shape[0])
#     self.v_out = (self.critic,)
#
#   def forward(self, inp):
#     x = inp.vis.type(torch.float32)
#     x = x / 255 - 0.5
#     x = self.conv(x)
#     x = x.view(x.size(0), -1)
#     h = fu.leaky_relu(self.lin1(torch.cat((x, inp.s), -1)))
#     v = self.critic(h)
#     a = self.actor(h)
#     return (a, None), tuple(v[:, i:i + 1] for i in range(self.num_critics))
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
