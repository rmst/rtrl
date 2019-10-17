from random import randint


class SimpleMemory:
  keep_reset_transitions: int = 0

  def __init__(self, memory_size, batchsize, device):
    self.device = device
    self.batchsize = batchsize
    self.capacity = memory_size
    self.memory = []  # list is much faster to index than deque for big sizes

    self.last_observation = None
    self.last_action = None

  def append(self, r, done, info, obs, action):
    if self.last_observation is not None:
      # info["reset"] = True means the episode reset shouldn't be learned (e.g. time limit)
      if self.keep_reset_transitions or not info.get('TimeLimit.truncated', False):
        self.memory.append((self.last_observation, self.last_action, r, obs, done))

    self.last_observation = obs
    self.last_action = action

    # remove old entries if necessary (delete generously so we don't have to do it often)
    if len(self.memory) > self.capacity:
      del self.memory[:self.capacity // 100]
    return self

  def __len__(self):
    return self.memory.__len__()

  def sample_indices(self):
    return (randint(0, len(self.memory) - 1) for _ in range(self.batchsize))

  def sample(self, indices=None):
    indices = self.sample_indices() if indices is None else indices
    batch = [self.memory[idx] for idx in indices]
    batch = collate(batch, self.device, non_blocking=True)
    return batch


# TODO: clean up
import numpy as np
from collections import Mapping, Sequence

import torch


numpy_type_map = {
  'float64': torch.FloatTensor,  # hack
  'float32': torch.FloatTensor,
  'float16': torch.HalfTensor,
  'int64': torch.LongTensor,
  'int32': torch.IntTensor,
  'int16': torch.ShortTensor,
  'int8': torch.CharTensor,
  'uint8': torch.ByteTensor,
  'bool': torch.FloatTensor,  # hack
}


def collate(batch, device=None, non_blocking=False):
  if not isinstance(batch, (tuple, list)):
    batch = tuple(batch)

  elem = batch[0]
  if isinstance(elem, torch.Tensor):
    # return torch.stack(batch, 0).to(device, non_blocking=non_blocking)
    if elem.numel() < 20000:  # TODO: link to the relavant profiling that lead to this threshold
      return torch.stack(batch).to(device, non_blocking=non_blocking)
    else:
      return torch.stack([b.contiguous().to(device, non_blocking=non_blocking) for b in batch], 0)
  elif isinstance(elem, np.ndarray):
    # if elem.size < 20000:  # TODO: link to the relavant profiling that lead to this threshold
    #   return torch.from_numpy(np.stack(batch)).to(device, non_blocking=non_blocking)
    # else:
    #   return torch.stack([torch.from_numpy(b).to(device, non_blocking=non_blocking) for b in batch], 0)
    try:
      return collate(tuple(torch.from_numpy(b) for b in batch), device, non_blocking)
    except TypeError:
      raise
  elif hasattr(elem, '__torch_tensor__'):
    return torch.stack([b.__torch_tensor__().to(device, non_blocking=non_blocking) for b in batch], 0)
  elif type(elem).__module__ == 'numpy' and elem.shape == ():
    py_type = float if elem.dtype.name.startswith('float') else int
    return numpy_type_map[elem.dtype.name](tuple(map(py_type, batch))).to(device, non_blocking=non_blocking)
  elif isinstance(elem, int):
    return torch.LongTensor(batch).to(device, non_blocking=non_blocking)
  elif isinstance(elem, float):
    return torch.FloatTensor(batch).to(device, non_blocking=non_blocking)  # hack
  elif isinstance(elem, bool):
    return torch.FloatTensor(batch).to(device, non_blocking=non_blocking)
  elif isinstance(elem, str):
    return batch
  elif isinstance(elem, Sequence):
    transposed = zip(*batch)
    return type(elem)(collate(samples, device, non_blocking) for samples in transposed)
  elif isinstance(elem, Mapping):
    return type(elem)((key, collate(tuple(d[key] for d in batch), device, non_blocking)) for key in elem)
  raise TypeError()


def partition(x):
  if isinstance(x, torch.Tensor):
    # return x.cpu()
    return x.cpu().numpy()  # perhaps we should convert this to tuple for consistency?
  elif isinstance(x, Mapping):
    m = {k: partition(x[k]) for k in x}
    numel = len(tuple(m.values())[0])
    out = tuple(type(x)((key, value[i]) for key, value in m.items()) for i in range(numel))
    return out

  raise TypeError()
