"""A simple command line interface"""

import sys

from rtrl.run import spec, init, run, init_and_run

cmd, args = sys.argv
if cmd == "spec":
  spec(eval(args[0]), args[1])
elif cmd == "init":
  init(*args)
elif cmd == "run":
  run(*args)
elif cmd == "init_and_run":
  init_and_run(*args)