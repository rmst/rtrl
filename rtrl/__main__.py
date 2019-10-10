"""A simple command line interface

Yaml is used to interpret the run specification because it doesn't require quoting around strings (compared to Python or Json).
Requiring quotes would be annoying because they are consumed by bash itself.
"""

import sys

from rtrl import *

_, cmd, *args = sys.argv

if cmd == "run":
  run(partial_from_dict(yaml.safe_load(args[0])), *args[1:])
elif cmd == "run_fs":
  run_fs(args[0], partial_from_dict(yaml.safe_load(args[1])))
elif cmd == "run_wandb":
  run_wandb(args[0], args[1], args[2], partial_from_dict(yaml.safe_load(args[3])), *args[4:])
else:
  raise AttributeError("Undefined command: " + cmd)
