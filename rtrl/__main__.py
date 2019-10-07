"""A simple command line interface"""

import sys

from rtrl import spec, init, resume, run

_, cmd, *args = sys.argv

if cmd == "spec":
  spec(eval(args[0]), args[1])
elif cmd == "init":
  init(*args)
elif cmd == "resume":
  resume(*args)
elif cmd == "run":
  run(*args)
else:
  raise AttributeError("Undefined command: " + cmd)
