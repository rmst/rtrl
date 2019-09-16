import functools
import inspect
import io
import json
import pickle
import weakref
from dataclasses import is_dataclass, dataclass, make_dataclass, fields, Field
from importlib import import_module
from itertools import chain
from typing import TypeVar, Union, Type, Callable, Any
from weakref import WeakKeyDictionary

import pandas as pd
import torch

T = TypeVar('T')  # helps with type inference in some editors


def pandas_dict(*args, **kwargs) -> pd.Series:
  return pd.Series(dict(*args, **kwargs), dtype=object)


def shallow_copy(obj: T) -> T:
  x = type(obj).__new__(type(obj))
  vars(x).update(vars(obj))
  return x


# noinspection PyPep8Naming
class cached_property:
  """Similar to `property` but after calling the getter/init function the result is cached.
  It can be used to create object attributes that aren't stored in the object's __dict__. """

  def __init__(self, init=None):
    self.cache = {}
    self.init = init

  def __get__(self, instance, owner):
    if id(instance) not in self.cache:
      if self.init is None: raise AttributeError()
      self.__set__(instance, self.init(instance))
    return self.cache[id(instance)][0]

  def __set__(self, instance, value):
    # Cache the attribute value based on the instance id. If instance is garbage collected its cached value is removed.
    self.cache[id(instance)] = (value, weakref.ref(instance, functools.partial(self.cache.pop, id(instance))))


# === partial ==========================================================================================================
def default():
  raise ValueError("This is a dummy function and not meant to be called.")


def partial(func: Type[T] = default, *args, **kwargs) -> Union[T, Type[T]]:
  """Like `functools.partial`, except if used as a keyword argument for another `partial` and no function is supplied.
   Then, the outer `partial` will insert the appropriate default value as the function. E.g. see `specs.py`. """

  for k, v in kwargs.items():
    if isinstance(v, functools.partial) and v.func is default:
      kwargs[k] = partial(inspect.signature(func).parameters[k].default, *v.args, **v.keywords)
  return functools.partial(func, *args, **kwargs)


def partial_to_dict(p: functools.partial, version="1"):
  assert not p.args, "So far only keyword arguments are supported, here"
  fields = {k: v.default for k, v in inspect.signature(p.func).parameters.items()}
  fields = {k: v for k, v in fields.items() if v is not inspect.Parameter.empty}
  diff = p.keywords.keys() - fields.keys()
  assert not diff, f"There are invalid keywords present: {diff}"
  fields.update(p.keywords)
  nested = {k: partial_to_dict(partial(v), version="") for k, v in fields.items() if callable(v)}
  simple = {k: v for k, v in fields.items() if k not in nested}
  output = {"__func__": p.func.__module__ + ":" + p.func.__qualname__, **simple, **nested}
  return dict(output, __format_version__=version) if version else output


def partial_from_dict(d: dict):
  d = d.copy()
  assert d.pop("__format_version__", "1") == "1"
  d = {k: partial_from_dict(v) if isinstance(v, dict) and "__func__" in v else v for k, v in d.items()}
  module, name = d.pop("__func__").split(":")
  func = getattr(import_module(module), name)
  return partial(func, **d)


# === serialization ====================================================================================================
def dump(obj, path):
  with open(path, 'wb') as f:
    return pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load(path):
  with open(path, 'rb') as f:
    return pickle.load(f)


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