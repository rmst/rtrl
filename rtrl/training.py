from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

import rtrl.sac
from pandas import DataFrame, Timestamp

from rtrl.testing import Test
from rtrl.util import pandas_dict, cached_property
from rtrl.wrappers import StatsWrapper
from rtrl.envs import GymEnv


@dataclass
class Training:
  Env: type = GymEnv
  Test: type = Test
  Agent: type = rtrl.sac.Agent
  steps: int = 1000  # number of steps per round
  rounds: int = 20  # number of rounds per epoch
  seed: int = 0
  epochs: int = 50

  # we use cached_property because we don't want to save these attributes to file
  env = cached_property(lambda self: StatsWrapper(self.Env(seed_val=self.seed + self.epoch), window=self.steps))
  last_transition = cached_property(lambda self: (None, 0., True, dict(reset=True)))

  def __post_init__(self):
    self.epoch = 0
    # print("Environment:", self.env)
    # noinspection PyArgumentList
    self.agent = self.Agent(self.env.observation_space, self.env.action_space)
    self.time = pd.Timedelta(0)

  def run_epoch(self):
    stats = []

    for rnd in range(self.rounds):
      print(f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, '=') + f" round {rnd}/{self.rounds} ".ljust(50, '='))
      t0 = pd.Timestamp.utcnow()
      stats_training = []

      # test runs in parallel to the training process
      # noinspection PyArgumentList
      test = self.Test(
        Env=self.Env,
        actor=deepcopy(self.agent.model),
        steps=self.steps * self.rounds,
        base_seed=self.seed + self.epochs
      )

      for step in range(self.steps):
        obs, r, done, info = self.last_transition

        if done:
          obs = self.env.reset()

        action, __ = self.agent.act(obs, r, done, info, train=True)
        stats_training.append(__)

        self.last_transition = self.env.step(action)

      stats += pandas_dict(**self.env.stats(),
                           time=self.time, round_time=Timestamp.utcnow() - t0,
                           **test.stats().add_suffix("_test"),
                           **DataFrame(stats_training).mean(skipna=True)),  # appending to stats

      self.time += Timestamp.utcnow() - t0

      print(stats[-1].add_prefix("  ").to_string(), '\n')

    self.epoch += 1
    return stats
