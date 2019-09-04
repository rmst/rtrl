import json
import os
import pickle
import sys
import shutil
import io
import torch

__version__ = "1"


def default_reduce(obj):
  """We manually reduce the object's class via its name."""
  assert obj.__class__.__name__ == obj.__class__.__qualname__, f"{obj.__class__} needs to be top-level"
  module = obj.__class__.__module__
  cls_name = obj.__class__.__name__
  state = obj.__getstate__() if hasattr(obj, "__getstate__") else vars(obj)
  return default_instantiate, (module, cls_name), state


def default_instantiate(module: str, cls_name: str):
  """In the future we could remap class names here, in case they have changed"""
  cls = getattr(sys.modules[module], cls_name)
  return cls.__new__(cls)


def dump(obj, path):
  """Like `pickle.dump`, except if `obj.__split_state__ == True`. Then the object's components (as returned by `__getstate__`) are being saved into different files.

  If `obj.__split_state__ == True`, sub-objects aren't deduplicated as usual (although this could be implemented via symlinks in the future).
  """

  if not getattr(obj, "__split_state__", False):
    # Note: Using `torch.save(obj, path)` seems no longer necessary, see https://blog.dask.org/2018/07/23/protocols-pickle.
    with open(path, 'wb') as f:
      return pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

  if os.path.isfile(path):
    os.remove(path)
  elif os.path.isdir(path) and os.path.exists(os.path.join(path, "__meta__")):
    shutil.rmtree(path)
  os.mkdir(path)

  creator_fn, creator_args, state = obj.__reduce__() if hasattr(obj, "__reduce__") else default_reduce(obj)
  assert "__meta__" not in state, '"__meta__" is a reserved name and can not be used as a state key'
  files = dict(state, __meta__=(__version__, (creator_fn, creator_args)))
  [dump(v, os.path.join(path, k)) for k, v in files.items()]


def load(path):
  """Like `pickle.load`, except if `path` points to a directory. Then it attempts to reassemble an object from the directory contents."""

  if not os.path.isdir(path):
    with open(path, 'rb') as f:
      return pickle.load(f)

  version, meta = load(os.path.join(path, "__meta__"))
  assert version == "1", f"Can't load {path}. Incompatible format version {version}."
  creator_fn, creator_args = meta
  obj = creator_fn(*creator_args)
  state = {k: load(os.path.join(path, k)) for k in os.listdir(path) if k != "__meta__"}
  obj.__setstate__(state) if hasattr(obj, "__setstate__") else vars(obj).update(state)
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


class Directory(dict):
  __split_state__ = True

  def __getstate__(self):
    return dict(self, __version__="1")

  def __setstate__(self, state):
    version = state.pop("__version__")
    assert version == "1", "Incompatible format version."
    self.update(state)


def test_save_split_dict():
  from tempfile import mkdtemp

  path = mkdtemp() + '/test'
  try:
    d = Directory(dict(a=3, b="fjdks"))
    dump(d, path)

    e = load(path)
    assert d == e
    print("success!")

  finally:
    shutil.rmtree(path)


if __name__ == "__main__":
  test_save_split_dict()
