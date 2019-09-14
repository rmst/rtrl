import functools
import inspect
import json
import os
import pickle
import time
from copy import deepcopy
from itertools import chain
from os.path import join, exists
from tempfile import mkdtemp
from typing import Union

from rtrl.lazyload import dump, load, save_json, load_json
from rtrl.training import Training
import sys
import torch
import rtrl.models
import gym.spaces
import pandas as pd
from rtrl.specs import *

from rtrl.util import partial, partial_to_dict, partial_from_dict

import yaml


def exec_cmd(_, cmd, *args):
  """A simple command line interface
  Usage: exec_cmd(*sys.argv), see __main__.py"""
  if cmd == "spec":
    spec(eval(args[0]), args[1])
  elif cmd == "init":
    init(*args)
  elif cmd == "run":
    run(*args)
  elif cmd == "make_and_run":
    make_and_run(*args)


def spec(run_cls: type, spec_path):
  """Create a spec json file from a subclass of Training or a partial (reconfigured class). See `specs.py` for examples."""
  run_cls = partial(run_cls)
  save_json(partial_to_dict(run_cls), spec_path)
  return spec_path


def init(spec_path, path):
  """Create a Training instance from a spec json file"""
  run_cls = partial_from_dict(load_json(spec_path))
  print("=== specification ".ljust(70, "="))
  print(yaml.dump(partial_to_dict(run_cls), indent=3, default_flow_style=False, sort_keys=False), end="")
  run_instance: Training = run_cls()
  dump(run_instance, path)


def run(path: str):
  """Load a Training instance and continue running it until the final epoch."""
  while True:
    run_instance: Training = load(path)
    run_instance.run_epoch()
    dump(run_instance, path)
    print("")
    time.sleep(1)  # on network file systems writing files is asynchronous and we need to wait for sync
    if run_instance.epoch == run_instance.epochs:
      break


def make_and_run(spec_path: str, run_path: str):
  if not exists(run_path):
    init(spec_path, run_path)
    print("")
  else:
    print("\n\n\n\n" + "Continuing to run..." + "\n\n")
  run(run_path)


# === tests ============================================================================================================

def test_spec_make_run():
  path = mkdtemp()
  print("="*70 + "\n")
  print("Running in:", path)
  print("")
  try:
    make_and_run(spec(MjTest, join(path, "spec.json")), join(path, "state"))
  finally:
    import shutil
    shutil.rmtree(path)


if __name__ == "__main__":
  test_spec_make_run()
