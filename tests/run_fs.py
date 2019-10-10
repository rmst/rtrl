"""
Test for scripts/rtrl-parallel
"""

import sys
from os import environ, mkdir
from os.path import dirname
from shutil import rmtree
from subprocess import check_call
from tempfile import mkdtemp
from rtrl import SimpleTraining, spec, save_json

ROOT = dirname(dirname(__file__))


def callx(args):
  print("$", *args)
  check_call(args)


if __name__ == "__main__":
  path = mkdtemp()
  try:
    print("=" * 70 + "\n")
    print("Running in:", path)
    print("")
    environ["EXPERIMENTS"] = path
    environ["PATH"] = dirname(sys.executable) + ":" + environ["PATH"]
    mkdir(path + "/e1")
    save_json(spec(SimpleTraining), path + "/e1/spec.json")
    try:
      callx(["rtrl-parallel", 'e1'])
    finally:
      callx(["ls", path])
      callx(["ls", path + "/e1"])
      callx(["cat", path + "/e1/output.txt"])
      print("=" * 70 + "\n")
  finally:
    rmtree(path)

