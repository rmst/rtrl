import atexit
import multiprocessing as mp
from dataclasses import dataclass
from functools import partial

import pandas as pd
from pandas import DataFrame

from rtrl.util import pandas_dict
from rtrl.wrappers import StatsWrapper


class Test:
  def __init__(self, number: int = 2, **kwargs):
    # Note: It is important that we `spawn` here. Using the default `fork`, will cause Pytorch 1.2 to lock up because it uses a buggy OpenMPI implementation (libgomp). Olexa Bilaniuk at Mila helped us figure this out.
    self.pool = mp.get_context('spawn').Pool(number)
    self.result_handle = self.pool.map_async(partial(run_test, **kwargs), range(number))

  def stats(self):
    st = self.result_handle.get()
    st = DataFrame(st)
    means = st.mean(skipna=True)
    # stds = st.std(skipna=True).add_suffix("std")
    return means

  def __del__(self):
    self.pool.close()
    self.pool.join()


def run_test(number, *, Env, actor, base_seed, steps):
  t0 = pd.Timestamp.utcnow()
  env = Env(seed_val=base_seed + number)
  with StatsWrapper(env, window=steps) as env:
    for step in range(steps):
      action, stats = actor.act(*env.transition)
      # action = env.action_space.sample()
      env.step(action)

    return pandas_dict(env.stats(), round_time=pd.Timestamp.utcnow() - t0)
