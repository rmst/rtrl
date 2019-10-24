from dataclasses import dataclass, InitVar

import gym
import torch
from rtrl.util import collate, partition
from rtrl.nn import TanhNormalLayer, SacLinear, big_conv
from torch.nn import Module, Linear, Sequential, ReLU, Conv2d, LeakyReLU
from torch.nn.functional import leaky_relu

from rtrl.sac_models import ActorModule


class Mlp(ActorModule):
  def __init__(self, observation_space, action_space, hidden_units: int = 256):
    super().__init__()
    assert isinstance(observation_space, gym.spaces.Tuple)
    input_dim = sum(s.shape[0] for s in observation_space)
    self.net = Sequential(
      SacLinear(input_dim, hidden_units), ReLU(),
      SacLinear(hidden_units, hidden_units), ReLU(),
    )
    self.critic = Linear(hidden_units, 1)
    self.actor_output_layer = TanhNormalLayer(hidden_units, action_space.shape[0])
    self.critic_output_layers = (self.critic,)

  def actor(self, x):
    return self[0]

  def forward(self, x):
    assert isinstance(x, tuple)
    x = torch.cat(x, dim=1)
    h = self.net(x)
    v = self.critic(h)
    action_distribution = self.actor_output_layer(h)
    return action_distribution, (v,)


class MlpDouble(ActorModule):
  def __init__(self, observation_space, action_space, hidden_units: int = 256):
    super().__init__()
    self.a: Mlp = Mlp(observation_space, action_space, hidden_units=hidden_units)
    self.b: Mlp = Mlp(observation_space, action_space, hidden_units=hidden_units)
    self.critic_output_layers = self.a.critic_output_layers + self.b.critic_output_layers

  def actor(self, x):
    return self.a(x)[0]

  def forward(self, x):
    action_distribution, (v0,) = self.a(x)
    _, (v1,) = self.b(x)
    return action_distribution, (v0, v1)


class ConvDouble(ActorModule):
  def __init__(self, observation_space, action_space, hidden_units: int = 256):
    super().__init__()
    self.a: ConvRTAC = ConvRTAC(observation_space, action_space, hidden_units=hidden_units)
    self.b: ConvRTAC = ConvRTAC(observation_space, action_space, hidden_units=hidden_units)
    self.critic_output_layers = self.a.critic_output_layers + self.b.critic_output_layers

  def actor(self, x):
    return self.a(x)[0]

  def forward(self, x):
    action_distribution, (v0,) = self.a(x)
    _, (v1,) = self.b(x)
    return action_distribution, (v0, v1)


class ConvRTAC(Module):
  def __init__(self, observation_space, action_space, hidden_units: int = 256):
    super().__init__()
    assert isinstance(observation_space, gym.spaces.Tuple)
    (img_sp, vec_sp), ac_sp = observation_space

    self.conv = big_conv(img_sp.shape[0])

    with torch.no_grad():
      conv_size = self.conv(torch.zeros((1, *img_sp.shape))).view(1, -1).size(1)

    self.lin1 = Linear(conv_size + vec_sp.shape[0] + ac_sp.shape[0], hidden_units)
    self.critic = Linear(hidden_units, 1)
    self.actor = TanhNormalLayer(hidden_units, action_space.shape[0])
    self.critic_output_layers = (self.critic,)

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