import functools
import inspect
import json
import os
import pickle
from copy import deepcopy
from itertools import chain
from os.path import join
from tempfile import mkdtemp
from rtrl.serialization import dump, load, save_json, load_json
from rtrl.subclass import conf, conf_to_dict
from rtrl.training import Training
import sys
import torch
import rtrl.models
import gym.spaces
import pandas as pd
from rtrl.configurations import *

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
    make_and_run(eval(args[0]), args[1])


def spec(run_cls: type, spec_path):
  """Create a spec json file from a subclass of Training or a partial (reconfigured class). See `configurations.py` for examples."""
  run_cls = partial(run_cls)
  save_json(partial_to_dict(run_cls), spec_path)


def init(spec_path, path):
  """Create a Training instance from a spec json file"""
  run_cls = partial_from_dict(load_json(spec_path))
  print(yaml.dump(dict(config=partial_to_dict(run_cls)), indent=2, default_flow_style=False, sort_keys=False))
  run_instance: Training = run_cls()
  dump(run_instance, path)


def run(path):
  """Load a Training instance and continue running it until the final epoch."""
  while True:
    run_instance: Training = load(path)
    run_instance.run_epoch()
    dump(run_instance, path)
    print("")
    if run_instance.epoch == run_instance.epochs:
      break


def make_and_run(run_cls, path):
  spec(run_cls, join(path, "spec.json"))
  init(join(path, "spec.json"), join(path, "run"))
  print("")
  run(join(path, "run"))


if __name__ == "__main__":
  path = mkdtemp()
  print("="*70)
  print("Running in:", path)
  print("")
  try:
    make_and_run(MjTest, path)
  finally:
    import shutil
    shutil.rmtree(path)
