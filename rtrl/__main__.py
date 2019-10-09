"""A simple command line interface"""

import sys

from rtrl import *

_, cmd, conf, *args = sys.argv
run_cls = partial_from_dict(yaml.safe_load(conf))

if cmd == "run":
  run(run_cls, *args)
elif cmd == "run_fs":
  run_fs(run_cls, *args)
elif cmd == "run_wandb":
  run_wandb(run_cls, *args)
else:
  raise AttributeError("Undefined command: " + cmd)
