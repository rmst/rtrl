import pickle
import os
from os.path import join, isdir, isfile, exists


class File:
  def __init__(self, path):
    self._path = path

  def __fspath__(self):
    """Allows File instances to be used with e.g. `open` and functions in the `os` package."""
    return self._path

  def __getitem__(self, item):
    path = join(self._path, item)
    if exists(path):
      return File(path)
    else:
      raise AttributeError(f"{path} does not exist.")

  def __getattr__(self, item):
    return self[item]

  def __call__(self):
    assert isfile(self)
    with open(self, "rb") as f:
      try:
        # print(f"loading {self._path}")
        return pickle.load(f)
      except pickle.UnpicklingError:
        raise AttributeError(f"Can't unpickle {self._path}")

  def __iter__(self):
    return (File(join(self._path, item)) for item in os.listdir(self))

  def __dir__(self):
    suggestions = list(super().__dir__())
    if isdir(self):
      suggestions += (f for f in os.listdir(self) if f.isidentifier())
    return suggestions

  def _ipython_key_completions_(self):
    """https://ipython.readthedocs.io/en/stable/config/integrating.html"""
    suggestions = []
    if isdir(self):
      suggestions += os.listdir(self)
    return suggestions

  def __repr__(self):
    return f"{self.__class__.__name__}({self._path})"


home = File(os.environ.get("HOME", "/"))
root = File("/")
