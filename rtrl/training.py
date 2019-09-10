from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

import rtrl.sac
from pandas import DataFrame, Timestamp

from rtrl.serialization import LazyLoad
from rtrl.testing import Test
from rtrl.util import pandas_dict, lazy_property
from rtrl.wrappers import StatsWrapper
from rtrl.envs import GymEnv


@dataclass
class Training(LazyLoad):
  Env: type = GymEnv
  Test: type = Test
  Agent: type = rtrl.sac.Agent
  steps: int = 100  # number of steps per round
  rounds: int = 5  # number of rounds per epoch
  seed: int = 0
  epochs: int = 50

  # we use lazy_property because we don't want to save the following properties to file
  env = lazy_property(lambda self: StatsWrapper(self.Env(seed_val=self.seed + self.epoch), window=self.steps))
  last_transition = lazy_property(lambda self: (None, 0., True, dict(reset=True)))

  def __post_init__(self):
    self.epoch = 0
    # print("Environment:", self.env)

    # noinspection PyArgumentList
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

      self.time += Timestamp.now() - t0
      stats += pandas_dict(**self.env.stats(),
                           **test.stats().add_prefix("test_"),
                           **DataFrame(stats_training).mean(skipna=True),
                           time=self.time, round_time=Timestamp.now() - t0),  # appending to stats

      print(stats[-1].add_prefix("  ").to_string(), '\n')

    self.stats = self.stats.append(stats, ignore_index=True, sort=True)  # concat with stats from previous episodes
    self.epoch += 1

  # def __getstate__(self):
  #   x: Training = shallow_copy(self)
  #   del x.env, x.last_transition  # not saving that for now
  #   del x.agent, x.stats
  #   return dict(vars=vars(x), agent=self.agent, stats=self.stats, version="1")
  #
  # def __setstate__(self, state):
  #   version = state.pop("version")
  #   assert version == "1", "Incompatible format version"
  #
  #   self.agent = state.pop("agent")
  #   self.stats = state.pop("stats")
  #   vars(self).update(state.pop("vars"))
  #   self.env = StatsWrapper(self.Env(seed=self.seed + self.epoch), window=self.steps)
  #   self.last_transition = None, np.asarray(0., np.float32), True, dict(reset=True)


