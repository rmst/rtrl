import os
from copy import deepcopy
from tempfile import mkdtemp

from rtrl.serialization import dump, load
from rtrl.training import Training

import sys

import torch
import rtrl.models
import gym.spaces


if __name__ == '__main__':
  # print(sys.argv)
  _, *args = sys.argv

  from rtrl import *

  if args:
    path, exp_str = args
    Exp = None
    try:
      Exp = eval(exp_str)
    finally:
      print(f"Exp = eval('{exp_str}') \n    = {Exp}", )
  else:
    base_path = mkdtemp()
    path = base_path + "/test"
    Exp = Training

  if os.path.exists(path):
    exp: Training = load(path)
  else:
    exp: Training = Exp()
    dump(exp, path)

  print("")

  try:
    while exp.epoch < exp.epochs:
      exp = load(path)
      exp.run_epoch()
      dump(exp, path)
      print("")
  except KeyboardInterrupt:
    pass
  finally:
    if not args:
      import shutil
      shutil.rmtree(base_path)