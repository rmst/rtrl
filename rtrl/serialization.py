import json
import os
import pickle
import sys
import shutil
import io
from importlib import import_module
from itertools import chain
from os.path import join, exists
from time import time

import torch

from rtrl.util import external_property


class LazyLoad:
  # we use lazy_property here because we don't want to save those properties to file
  _lazyload_path = external_property(lambda s: None)
  _lazyload_timestamp = external_property(lambda s: None)
  _lazyload_files = external_property(lambda s: set())

  def __getattribute__(self, item):
    # looking for item in __dict__
    d = object.__getattribute__(self, "__dict__")
    if item in d:
      return d[item]

    # looking to load item from files
    f = object.__getattribute__(self, "_lazyload_files")
    if item in f:
      path = join(self._lazyload_path, item)
      assert os.path.getmtime(path) <= self._lazyload_timestamp, f"{path} changed after object creation"  # we currently don't check nested LazyLoad objects
      v = self.__dict__[item] = load(path)
      setattr(self, item, v)
      return v

    # looking for item in class
    return super().__getattribute__(item)

  def __dir__(self):
    return chain(super().__dir__(), self._lazyload_files)


def dump(obj, path):
  """Like `pickle.dump`. If `obj` is an instance of `LazyLoad` its components are saved as individual files such that they can be loaded lazily later.
  """

  if not isinstance(obj, LazyLoad):
    # Note: Using `torch.save(obj, path)` seems no longer necessary, see https://blog.dask.org/2018/07/23/protocols-pickle.
    with open(path, 'wb') as f:
      return pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

  if os.path.isfile(path):
    os.remove(path)
  elif os.path.isdir(path) and exists(join(path, "__meta__.json")):
    shutil.rmtree(path)
  os.mkdir(path)

  simple = {k: v for k, v in vars(obj).items() if isinstance(v, (float, int, str, bool))}
  other = {k: v for k, v in vars(obj).items() if k not in simple}
  save_json(dict(module=obj.__class__.__module__, cls=obj.__class__.__qualname__, version="1", __dict__=simple), join(path, "__meta__.json"))
  [dump(v, os.path.join(path, k)) for k, v in other.items()]


def load(path):
  """Like `pickle.load`, except if `path` points to a directory. Then it attempts to reassemble an object from the directory contents."""

  if not os.path.isdir(path):
    with open(path, 'rb') as f:
      return pickle.load(f)

  meta = load_json(os.path.join(path, "__meta__.json"))
  assert meta['version'] == "1", f"Can't load {path}. Incompatible format version {meta['version']}."
  files = set(os.listdir(path))
  files.remove("__meta__.json")
  cls = getattr(import_module(meta["module"]), meta["cls"])
  obj: LazyLoad = cls.__new__(cls)
  obj.__dict__.update(meta["__dict__"])
  obj._lazyload_path = path
  obj._lazyload_files = files
  obj._lazyload_timestamp = time()
  return obj


# === Utilities and Testing ============================================================================================

def dumps_torch(obj):
  with io.BytesIO() as f:
    torch.save(obj, f)
    return f.getvalue()


def loads_torch(b: bytes):
  with io.BytesIO(b) as f:
    return torch.load(f)


def save_json(d, path):
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2)


def load_json(path):
  with open(path, 'r', encoding='utf-8') as f:
    return json.load(f)



# def test_save_split_dict():
#   from tempfile import mkdtemp
#
#   path = mkdtemp() + '/test'
#   try:
#     d = Directory(dict(a=3, b="fjdks"))
#     dump(d, path)
#
#     e = load(path)
#     assert d == e
#     print("success!")
#
#   finally:
#     shutil.rmtree(path)
#
#
# if __name__ == "__main__":
#   test_save_split_dict()
