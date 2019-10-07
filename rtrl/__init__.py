import time
from os.path import exists
from tempfile import mkdtemp

import pandas as pd
import yaml

from rtrl.training import Training
from rtrl.util import partial, save_json, partial_to_dict, partial_from_dict, load_json, dump, load


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
# ======================================================================================================================


def spec(run_cls: type, spec_path):
  """Create a spec json file from a subclass of Training or a partial (reconfigured class). See `specs.py` for examples."""
  run_cls = partial(run_cls)
  save_json(partial_to_dict(run_cls), spec_path)


def init(spec_path, path):
  """Create a Training instance from a spec json file"""
  run_cls = partial_from_dict(load_json(spec_path))
  print("=== specification ".ljust(70, "="))
  print(yaml.dump(partial_to_dict(run_cls), indent=3, default_flow_style=False, sort_keys=False), end="")
  run_instance: Training = run_cls()
  dump(run_instance, path+'/state')
  dump(pd.DataFrame(), path+"/stats")


def resume(path: str):
  """Load a Training instance and resume running it until the final epoch."""
  while True:
    time.sleep(1)  # on network file systems writing files is asynchronous and we need to wait for sync
    run_instance: Training = load(path+"/state")
    stats = run_instance.run_epoch()
    dump(load(path+"/stats").append(stats, ignore_index=True), path+"/stats")  # concat with stats from previous episodes
    dump(run_instance, path+"/state")
    print("")
    if run_instance.epoch == run_instance.epochs:
      break


def run(path: str):
  if not exists(path+'/state'):
    init(path+'/spec.json', path)
    print("")
  else:
    print("\n\n\n\n" + "Resuming..." + "\n\n")
  resume(path)


def spec_init_run(conf: type, run_path: str = None):
  path = mkdtemp()
  print("="*70 + "\n")
  print("Running in:", path)
  print("")
  try:
    spec(conf, path + "/spec.json")
    run(path)
  finally:
    import shutil
    shutil.rmtree(path)

  from rtrl import SimpleTraining, spec_init_run


# === tests ============================================================================================================
if __name__ == "__main__":
  spec_init_run(SimpleTraining)
