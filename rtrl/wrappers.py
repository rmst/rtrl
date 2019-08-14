from collections import Sequence, Mapping

import gym
import numpy as np


class DictObservationWrapper(gym.ObservationWrapper):
  def __init__(self, env, key='vector'):
    super().__init__(env)
    self.key = key
    self.observation_space = gym.spaces.Dict({self.key: env.observation_space})

  def observation(self, observation):
    return {self.key: observation}


class DictActionWrapper(gym.Wrapper):
  def __init__(self, env, key='value'):
    super().__init__(env)
    self.key = key
    self.action_space = gym.spaces.Dict({self.key: env.action_space})

  def step(self, action: dict):
    return self.env.step(action['value'])


def get_wrapper_by_class(env, cls):
  if isinstance(env, cls):
    return env
  elif isinstance(env, gym.Wrapper):
    return get_wrapper_by_class(env.env, cls)


from copy import deepcopy
from functools import partial
from uuid import uuid4

class_dict = {}


def subclass(cls, attributes, name=None):
  name = name or cls.__name__ + "_" + str(uuid4())[:5]
  global class_dict
  if name not in class_dict:
    attributes.update(__reduce__=reduce)
    class_dict[name] = type(name, (cls,), attributes)
    print("subclass", class_dict[name])
  return class_dict[name]


def subclass_and_instantiate(base_class, attributes, name):
  cls = subclass(base_class, attributes, name)
  return cls.__new__(cls)


def reduce(self):
  base, = self.__class__.__bases__  # self.__class__ has a single base class (we created it with subclass)
  attributes = {k: v for k, v in vars(self.__class__).items() if not k.startswith("__")}
  name = self.__class__.__name__
  state = self.__getstate__() if hasattr(self, "__getstate__") else vars(self)
  print("reduce", base, attributes, name, state)
  return subclass_and_instantiate, (base, attributes, name), state


class Subclassable:
  def __new__(cls, *args, **kwargs):
    if ... in args:
      assert len(args) == 1, 'currently no positional arguments allowed'
      obj = subclass(cls, kwargs)
      print("__new__", obj)
      return obj
    else:
      return super().__new__(cls, *args, **kwargs)


class A(Subclassable):
  __: ...


a = A()
B = A(..., b=3)
b = B()
b.c = 9
b2 = deepcopy(b)
b2.c, b2.b

class NormalizeActionWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.scale = env.action_space.high - env.action_space.low
    self.shift = env.action_space.low
    self.action_space = gym.spaces.Box(-np.ones_like(self.shift), np.ones_like(self.shift), dtype=env.action_space.dtype)

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)

  def step(self, action):
    action = action / 2 + 0.5  # 0 < a < 1
    action = action * self.scale + self.shift
    return self.env.step(action)


class TimeLimitResetWrapper(gym.Wrapper):
  """"""
  NO_LIMIT = 1 << 31

  def __init__(self, env, max_steps=None, key='reset'):
    super().__init__(env)
    self.reset_key = key
    from gym.wrappers import TimeLimit
    self.enforce = bool(max_steps)
    if max_steps is None:
      tl = get_wrapper_by_class(env, TimeLimit)
      max_steps = self.NO_LIMIT if tl is None else tl._max_episode_steps
      # print("TimeLimitResetWrapper.max_steps =", max_steps)

    self.max_steps = max_steps
    self.t = 0

  def reset(self, **kwargs):
    m = self.env.reset(**kwargs)
    self.t = 0
    return m

  def step(self, action):
    m, r, d, info = self.env.step(action)

    # we don't consider it a "true" terminal state if we ran out of time
    reset = (self.t == self.max_steps - 1) or info.get(self.reset_key, False)
    if not self.enforce:
      if reset:
        assert d, f"something went wrong t={self.t}, max_steps={self.max_steps}, info={info}"
    else:
      d = d or reset
    info = {**info, self.reset_key: reset}
    self.t += 1
    return m, r, d, info


def deepmap(f, m):
  """Example: deepmap({torch.Tensor: lambda t: t.detach()}, x)"""
  for cls in f:
    if isinstance(m, cls):
      return f[cls](m)
  if isinstance(m, Sequence):
    return type(m)(deepmap(f, x) for x in m)
  elif isinstance(m, Mapping):
    return type(m)((k, deepmap(f, m[k])) for k in m)
  else:
    raise AttributeError()


class Float64ToFloat32(gym.ObservationWrapper):
  """converts states and rewards to float32"""

  # TODO: change observation/action spaces to correct dtype
  def observation(self, observation):
    observation = deepmap({np.ndarray: self.convert}, observation)
    return observation

  def convert(self, x):
      return np.asarray(x, np.float32) if x.dtype == np.float64 else x

  def step(self, action):
    s, r, d, info = super().step(action)
    return s, np.asarray(r, np.float32), d, info