import os
import tempfile
import time
from os.path import exists
from tempfile import mkdtemp

import pandas as pd
import yaml

from rtrl.training import Training
from rtrl.util import partial, save_json, partial_to_dict, partial_from_dict, load_json, dump, load


def iterate_episodes(run_cls: type = Training, checkpoint_path: str = None):
  """Generator [1] yielding episode statistics (pd.DataFrame) while running and checkpointing
  - run_cls: can by any callable that outputs an appropriate run object (e.g. has a 'run_epoch' method)

  [1] https://docs.python.org/3/howto/functional.html#generators
  """
  checkpoint_path = checkpoint_path or tempfile.mktemp("_remove_on_exit")

  try:
    if not exists(checkpoint_path):
      print("=== specification ".ljust(70, "="))
      print(yaml.dump(partial_to_dict(run_cls), indent=3, default_flow_style=False, sort_keys=False), end="")
      run_instance = run_cls()
      dump(run_instance, checkpoint_path)
      print("")

    while True:
      time.sleep(1)  # on network file systems writing files is asynchronous and we need to wait for sync
      run_instance = load(checkpoint_path)
      yield run_instance.run_epoch()  # yield stats data frame (this makes this function a generator)
      print("")
      dump(run_instance, checkpoint_path)
      if run_instance.epoch == run_instance.epochs:
        break
  finally:
    if checkpoint_path.endswith("_remove_on_exit") and exists(checkpoint_path):
      os.remove(checkpoint_path)


def run(run_cls: type = Training, checkpoint_path: str = None):
  list(iterate_episodes(run_cls, checkpoint_path))


def run_wandb(run_cls: type, checkpoint_path: str, entity, project, run_id):
  """run and save config and stats to https://wandb.com"""
  import wandb
  wandb.init(entity=entity, project=project, resume=run_id, config=partial_to_dict(run_cls))
  for stats in iterate_episodes(run_cls, checkpoint_path):
    [wandb.log(dict(s)) for _, s in stats.T.items()]


def run_fs(run_cls: type, path: str):
  """run and save config and stats to `path` (with pickle)"""
  save_json(partial_to_dict(run_cls), path+'/spec.json')
  if not exists(path+'/stats'):
    dump(pd.DataFrame(), path+'/stats')
  for stats in iterate_episodes(run_cls, path + '/state'):
    dump(load(path+'/stats').append(stats, ignore_index=True), path+'/stats')  # concat with stats from previous episodes


# === specifications ===================================================================================================

MjTest = partial(
  Training,
  epochs=3,
  rounds=5,
  steps=10,
  Agent=partial(memory_size=1000000),
  Env=partial(id="Pendulum-v0"),
)

SimpleTraining = partial(
  Training,
  epochs=50,
  rounds=20,
  steps=1000,
  Agent=partial(memory_size=1000000),
  Env=partial(id="Pendulum-v0"),
)

MjTraining = partial(
  Training,
  epochs=50,
  rounds=20,
  steps=1000,
  Agent=partial(memory_size=1000000, batchsize=256, start_training=10000),
  Env=partial(id='Walker2d-v2'),
)


# === tests ============================================================================================================
if __name__ == "__main__":
  run(SimpleTraining)
