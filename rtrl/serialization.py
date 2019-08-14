import json
import os
import sys

import torch

import rtrl


def save(obj, path):
  if hasattr(obj, "__getfiles__"):
    os.mkdir(path)
    assert obj.__class__.__name__ == obj.__class__.__qualname__, f"Can only serialize top-level class objects and functions, not f{obj.__class__}"
    info = dict(rtrl_version=rtrl.__version__, module=obj.__class__.__module__, name=obj.__class__.__name__)
    with open(os.path.join(path, "__pyobj__.json"), 'w', encoding='utf-8') as f:
      json.dump(info, f, ensure_ascii=False, indent=2)
    for k, v in obj.__getfiles__().items():
      save(v, os.path.join(path, k))
  else:
    with open(path, "wb") as f:
      torch.save(obj, f)


def load(path):
  if os.path.isdir(path):
    files = os.listdir(path)
    files.remove("__pyobj__.json")
    with open(os.path.join(path, "__pyobj__.json"), 'r', encoding='utf-8') as f:
      info = json.load(f)
    cls = getattr(sys.modules[info["module"]], info["name"])
    obj = cls.__new__(cls)
    obj.__setfiles__({f: load(os.path.join(path, f)) for f in files})
    return obj
  else:
    with open(path, 'rb') as f:
      return torch.load(f)


class Directory(dict):
  def __getfiles__(self):
    return self

  def __setfiles__(self, files):
    self.update(files)


if __name__ == "__main__":
  import shutil
  shutil.rmtree("/home/simon/dfdks")
  d = Directory(dict(a=3, b="fjdks"))
  save(d, "/home/simon/dfdks")

  f = load('/home/simon/dfdks')
  print("done")
