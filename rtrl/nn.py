from copy import deepcopy
from dataclasses import InitVar, dataclass

import numpy as np
import torch
from torch.distributions import Distribution, Normal
from torch.nn.init import kaiming_uniform_, xavier_uniform_, calculate_gain

from rtrl import partial


def no_grad(model):
  for p in model.parameters():
    p.requires_grad = False
  return model


def copy_shared(model_a):
  """Create a deepcopy of a model but with the underlying state_dict shared. E.g. useful in combination with `no_grad`."""
  model_b = deepcopy(model_a)
  sda = model_a.state_dict(keep_vars=True)
  sdb = model_b.state_dict(keep_vars=True)
  for key in sda:
    a, b = sda[key], sdb[key]
    b.data = a.data  # strangely this will not make a.data and b.data the same object but their underlying data_ptr will be the same
    assert b.storage().data_ptr() == a.storage().data_ptr()
  return model_b


@dataclass(eq=0)
class PopArt:
  """PopArt https://arxiv.org/pdf/1809.04474.pdf"""
  dim: InitVar[int] = 1
  beta: float = 0.0003
  update_weights: int = 1  # i.e. should we try to preserve outputs. If no that's just a running mean, std

  def __post_init__(self, dim):
    self.m1 = torch.zeros((dim,))
    self.m2 = torch.ones((dim,))
    self.std = torch.ones((dim,))

  def to(self, device):
    [setattr(self, k, v.to(device)) for k, v in vars(self).items() if torch.is_tensor(v)]
    return self

  def update(self, targets):
    targets = targets.detach()
    self.m1_old = self.m1
    self.m2_old = self.m2
    self.std_old = self.std
    self.m1 = (1-self.beta) * self.m1_old + self.beta * targets.mean(0)
    self.m2 = (1-self.beta) * self.m2_old + self.beta * (targets * targets).mean(0)
    self.std = (self.m2 - self.m1 * self.m1).sqrt().clamp(0.0001, 1e6)

  def update_lin(self, lin):
    if not self.update_weights:
      return
    assert isinstance(lin, torch.nn.Linear)
    assert self.std.shape == (1,), 'this has only been tested with 1d outputs, verify that the following line is ' \
                                   'doing the right thing to the weight matrix, then remove this statement '
    lin.weight.data *= self.std_old / self.std
    lin.bias.data *= self.std_old
    lin.bias.data += self.m1_old - self.m1
    lin.bias.data /= self.std

  def normalize(self, x):
    return (x - self.m1) / self.std

  def unnormalize(self, x):
    return x * self.std + self.m1


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
    assert hasattr(x, "pre_tanh_value")
    assert x.dim() == 2 and x.pre_tanh_value.dim() == 2
    return self.normal.log_prob(x.pre_tanh_value) - torch.log(
      1 - x * x + self.epsilon
    )

  def sample(self, sample_shape=torch.Size()):
    z = self.normal.sample(sample_shape)
    out = torch.tanh(z)
    out.pre_tanh_value = z
    return out

  def rsample(self, sample_shape=torch.Size()):
    z = self.normal.rsample(sample_shape)
    out = torch.tanh(z)
    out.pre_tanh_value = z
    return out


# noinspection PyAbstractClass
class Independent(torch.distributions.Independent):
  def sample_deterministic(self):
    return torch.tanh(self.base_dist.normal_mean)


class TanhNormalLayer(torch.nn.Module):
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
    log_std = torch.clamp(log_std, -20, 2)
    std = torch.exp(log_std)
    # a = TanhTransformedDist(Independent(Normal(m, std), 1))
    a = Independent(TanhNormal(mean, std), 1)
    return a


# class TanhNormalLayer(torch.nn.Module):
#   def __init__(self, n, m):
#     super().__init__()
#     with torch.no_grad():
#       self.lin_mean = torch.nn.Linear(n, m)
#       xavier_uniform_(self.lin_mean.weight, calculate_gain('tanh'))
#       self.lin_mean.bias.fill_(0)
#       self.lin_std = torch.nn.Linear(n, m)
#       xavier_uniform_(self.lin_std.weight, calculate_gain('tanh'))
#       self.lin_std.bias.fill_(0)
#
#   def forward(self, x):
#     mean = self.lin_mean(x)
#     log_std = self.lin_std(x)
#     log_std = torch.clamp(log_std, -20, 2)
#     std = torch.exp(log_std)
#     # a = TanhTransformedDist(Independent(Normal(m, std), 1))
#     a = Independent(TanhNormal(mean, std), 1)
#     return a


class RlkitLinear(torch.nn.Linear):
  def __init__(self, *args):
    super().__init__(*args)
    # TODO: investigate the following
    # this mistake seems to be in rlkit too
    # https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/pytorch_util.py
    fan_in = self.weight.shape[0]  # this is actually fanout!!!
    bound = 1. / np.sqrt(fan_in)
    self.weight.data.uniform_(-bound, bound)
    self.bias.data.fill_(0.1)


class SacLinear(torch.nn.Linear):
  def __init__(self, in_features, out_features):
    super().__init__(in_features, out_features)
    with torch.no_grad():
      self.weight.uniform_(-0.06, 0.06)  # 0.06 == 1 / sqrt(256)
      self.bias.fill_(0.1)


class BasicReLU(torch.nn.Linear):
  def forward(self, x):
    x = super().forward(x)
    return torch.relu(x)


class AffineReLU(BasicReLU):
  def __init__(self, in_features, out_features, init_weight_bound: float = 1., init_bias: float = 0.):
    super().__init__(in_features, out_features)
    bound = init_weight_bound / np.sqrt(in_features)
    self.weight.data.uniform_(-bound, bound)
    self.bias.data.fill_(init_bias)


class NormalizedReLU(torch.nn.Sequential):
  def __init__(self, in_features, out_features, prenorm_bias=True):
    super().__init__(
      torch.nn.Linear(in_features, out_features, bias=prenorm_bias),
      torch.nn.LayerNorm(out_features),
      torch.nn.ReLU())


class KaimingReLU(torch.nn.Linear):
  def __init__(self, in_features, out_features):
    super().__init__(in_features, out_features)
    with torch.no_grad():
      kaiming_uniform_(self.weight)
      self.bias.fill_(0.)

  def forward(self, x):
    x = super().forward(x)
    return torch.relu(x)


Linear10 = partial(AffineReLU, init_bias=1.)
Linear04 = partial(AffineReLU, init_bias=0.4)
LinearConstBias = partial(AffineReLU, init_bias=0.1)
LinearZeroBias = partial(AffineReLU, init_bias=0.)
AffineSimon = partial(AffineReLU, init_weight_bound=0.01, init_bias=1.)


def dqn_conv(n):
  return torch.nn.Sequential(
      torch.nn.Conv2d(n, 32, kernel_size=8, stride=4),
      torch.nn.ReLU(),
      torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
      torch.nn.ReLU(),
      torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
      torch.nn.ReLU()
    )


def big_conv(n):
  return torch.nn.Sequential(
    torch.nn.Conv2d(n, 32, 8, stride=2), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(32, 32, 4, stride=2), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(32, 32, 4, stride=2), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(32, 32, 4, stride=1), torch.nn.LeakyReLU(),
  )