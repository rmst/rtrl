from os.path import join, dirname

with open(join(dirname(dirname(__file__)), "version"), 'r') as f:
  __version__ = f.read()
