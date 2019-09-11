import multiprocessing as mp
from dataclasses import dataclass

import pandas as pd
from pandas import DataFrame

from rtrl.util import shallow_copy, pandas_dict
from rtrl.wrappers import StatsWrapper


@dataclass
class Test:
  Env: type
  actor: object
  base_seed: int
  steps: int

  number: int = 4
  workers: int = 1

  def __post_init__(self):
    # Note: It is important that we `spawn` here. Using the default `fork`, will cause Pytorch 1.2 to lock up because it uses a buggy OpenMPI implementation (libgomp). Olexa Bilaniuk at Mila helped us figure this out.
    self.pool = mp.get_context('spawn').Pool(self.workers)
    self.result_handle = self.pool.map_async(self.run, range(self.number))

  def __getstate__(self):
    x = shallow_copy(self)
    del x.pool, x.result_handle  # instances of mp.Pool, etc. cannot be shared between processes
    return vars(x)

  def run(self, number=0):
    t0 = pd.Timestamp.now()
    env = self.Env(seed_val=self.base_seed + number)
    env = StatsWrapper(env, window=self.steps)

    obs, r, done, info = None, 0., True, {}
    for step in range(self.steps):
      if done:
        obs = env.reset()
      action, stats = self.actor.act(obs, r, done, info)
      # action = env.action_space.sample()
      obs, r, done, info = env.step(action)

    return pandas_dict(env.stats(), round_time=pd.Timestamp.now() - t0)

  def stats(self):
    st = self.result_handle.get()
    st = DataFrame(st)
    means = st.mean(skipna=True)
    # stds = st.std(skipna=True).add_suffix("std")
    self.pool.close()
    self.pool.join()
    return means
