from copy import deepcopy
from dataclasses import InitVar, dataclass

import torch


def no_grad(model):
  for p in model.parameters():
    p.requires_grad = False
  return model


@dataclass
class PopArt:
  """PopArt https://arxiv.org/pdf/1809.04474.pdf"""
  dim: InitVar
  device: InitVar
  beta: float = 0.0003
  update_weights: int = 1  # i.e. should we try to preserve outputs. If no that's just a running mean, std

  def __post_init__(self, dim, device):
    self.device = device
    self.m1 = torch.zeros((dim,)).to(device)
    self.m2 = torch.ones((dim,)).to(device)
    self.std = torch.ones((dim,)).to(device)

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


def copy_shared(model_a):
  model_b = deepcopy(model_a)
  sda = model_a.state_dict(keep_vars=True)
  sdb = model_b.state_dict(keep_vars=True)
  for key in sda:
    a, b = sda[key], sdb[key]
    b.data = a.data  # strangely this will not make a.data and b.data the same object but their underlying data_ptr will be the same
    assert b.storage().data_ptr() == a.storage().data_ptr()
  return model_b