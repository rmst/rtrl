"""A simple command line interface

Yaml is used to interpret the run specification because it doesn't require quoting around strings (compared to Python or Json).
Requiring quotes would be annoying because they are consumed by bash itself.
"""

import sys

from rtrl import *
from rtrl.util import partial_from_args

_, cmd, *args = sys.argv


def parse_args(func, *a):
  kwargs = dict(x.split("=") for x in a)
  return partial_from_args(func, kwargs)


if cmd == "run":
  run(parse_args(*args))
elif cmd == "run_fs":
  run_fs(args[0], parse_args(*args[1:]))
elif cmd == "run_wandb":
  run_wandb(args[0], args[1], args[2], args[3], parse_args(args[4:]))
else:
  raise AttributeError("Undefined command: " + cmd)
