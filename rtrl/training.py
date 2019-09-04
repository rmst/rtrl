from copy import deepcopy, copy
from dataclasses import dataclass, InitVar
from functools import partial
from time import time
import pandas as pd
import gym
import numpy as np
import torch

import rtrl.sac
from pandas import DataFrame, Timestamp

from rtrl.models import Mlp
from rtrl.serialization import Directory, load
from rtrl.util import apply_kwargs, shallow_copy, pandas_dict
from rtrl.wrappers import TimeLimitResetWrapper, DictObservationWrapper, NormalizeActionWrapper, DictActionWrapper, Float64ToFloat32, StatsWrapper
# import torch.multiprocessing as mp
import multiprocessing as mp


@dataclass
class GymEnv(gym.Wrapper):
  id: str = "Pendulum-v0"

  def __post_init__(self):
    env = gym.make(self.id)
    env = Float64ToFloat32(env)
    env = TimeLimitResetWrapper(env)
    env = DictObservationWrapper(env)
    assert isinstance(env.action_space, gym.spaces.Box)
    env = NormalizeActionWrapper(env)
    env = DictActionWrapper(env)
    super().__init__(env)

  def __del__(self):
    self.close()


@dataclass
class Test:
  Env: type
  actor: object
  base_seed: int
  steps: int
  seeds: int = 4
  workers: int = 1

  def __post_init__(self):
    self.pool = mp.get_context('spawn').Pool(self.workers)  # Note: It is important that we `spawn` here. Using the default `fork`, will cause Pytorch 1.2 to lock up because it uses a buggy OpenMPI implementation (libgomp). Olexa Bilaniuk at Mila helped us figure this out.
    self.result_handle = self.pool.map_async(self.run, range(self.seeds))

  def stats(self):
    st = self.result_handle.get()
    st = DataFrame(st)
    means = st.mean(skipna=True)
    # stds = st.std(skipna=True).add_suffix("std")
    self.pool.close()
    self.pool.join()
    return means

  def __getstate__(self):
    x = shallow_copy(self)
    del x.pool, x.result_handle  # instances of mp.Pool, etc. cannot be shared between processes
    return vars(x)

  def run(self, number=0):
    t0 = pd.Timestamp.now()
    env = self.Env(seed=self.base_seed+number)
    env = StatsWrapper(env, horizon=self.steps)

    obs, r, done, info = None, 0., True, {}
    for step in range(self.steps):
      if done:
        obs = env.reset()
      action, stats = self.actor.act(obs, r, done, info)
      # action = env.action_space.sample()
      obs, r, done, info = env.step(action)

    return pandas_dict(env.stats(), round_time=pd.Timestamp.now() - t0)


@dataclass
class Training:
  Env: type = GymEnv
  Test: type = Test
  Agent: type = rtrl.sac.Agent
  steps: int = 100  # number of steps per round
  rounds: int = 5  # number of rounds per epoch
  seed: int = 0
  epochs: int = 50

  last_transition = None

  def __post_init__(self):
    self.epoch = 0
    self.env = StatsWrapper(self.Env(seed=self.seed + self.epoch), horizon=self.steps)
    self.last_transition = None, 0., True, dict(reset=True)
    print(self.env)

    self.agent = self.Agent(self.env.observation_space, self.env.action_space)

    self.stats = DataFrame()
    self.time = pd.Timedelta(0)

  def run_epoch(self):
    t0 = pd.Timestamp.now()
    stats = []

    for rnd in range(self.rounds):
      print(f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, '='), f"round {rnd}/{self.rounds} ".ljust(50, '='))
      stats_training = []

      # test runs in parallel to the training process
      test = self.Test(
        Env=self.Env,
        actor=deepcopy(self.agent.model),
        steps=self.steps * self.rounds,
        base_seed=self.seed + self.epochs)

      for step in range(self.steps):
        obs, r, done, info = self.last_transition

        if done:
          obs = self.env.reset()

        action, __ = self.agent.act(obs, r, done, info, train=True)
        stats_training.append(__)

        self.last_transition = self.env.step(action)

      self.time += Timestamp.now() - t0
      stats += pandas_dict(**self.env.stats(),
                           **test.stats().add_prefix("test_"),
                           **DataFrame(stats_training).mean(skipna=True),
                           time=self.time, round_time=Timestamp.now() - t0),  # appending to stats

      print(stats[-1].add_prefix("  ").to_string(), '\n')

    self.stats = self.stats.append(stats, ignore_index=True, sort=True)  # concat with stats from previous episodes
    self.epoch += 1

  __split_state__ = True

  def __getstate__(self):
    x: Training = shallow_copy(self)
    del x.env, x.last_transition  # not saving that for now
    del x.agent, x.stats
    return dict(vars=vars(x), agent=self.agent, stats=self.stats, version="1")

  def __setstate__(self, state):
    version = state.pop("version")
    assert version == "1", "Incompatible format version"

    self.agent = state.pop("agent")
    self.stats = state.pop("stats")
    vars(self).update(state.pop("vars"))
    self.env = StatsWrapper(self.Env(seed=self.seed + self.epoch), horizon=self.steps)
    self.last_transition = None, np.asarray(0., np.float32), True, dict(reset=True)
