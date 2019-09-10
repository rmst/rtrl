import functools
import inspect
import weakref
from dataclasses import is_dataclass, dataclass, make_dataclass, fields, Field
from importlib import import_module
from itertools import chain
from typing import TypeVar, Union, Type, Callable, Any
from weakref import WeakKeyDictionary

import pandas as pd

T = TypeVar('T')


def pandas_dict(*args, **kwargs) -> pd.Series:
  return pd.Series(dict(*args, **kwargs), dtype=object)


def shallow_copy(obj: T) -> T:
  x = type(obj).__new__(type(obj))
  vars(x).update(vars(obj))
  return x


# === lazy properties ==================================================================================================
def lazy_property(init=None):
  cache = {}
  return property(functools.partial(get_cached, init, cache), functools.partial(set_cached, cache))


def del_cached(cache, obj_id, _):
  del cache[obj_id]


def set_cached(cache, obj, value):
  cache[id(obj)] = (value, weakref.ref(obj, functools.partial(del_cached, cache, id(obj))))


def get_cached(fun, cache, obj):
  if id(obj) not in cache:
    if fun is None: raise AttributeError()
    set_cached(cache, obj, fun(obj))
  return cache[id(obj)][0]


# === partial ==========================================================================================================
def default():
  raise ValueError("This is a dummy function not meant to be called. It can be used within a nested `partial` where it is replaced with the default value of a keyword argument.")


def partial(func: Type[T] = default, *args, **kwargs) -> Union[T, Type[T]]:
  """Like `functools.partial`. However, when used as a keyword argument within another `partial` and if `default` is supplied as the function, `default` will be replaced with the default value for that keyword argument."""
  for k, v in kwargs.items():
    if isinstance(v, functools.partial) and v.func is default:
      kwargs[k] = partial(inspect.signature(func).parameters[k].default, *v.args, **v.keywords)
  return functools.partial(func, *args, **kwargs)


def partial_to_dict(p: functools.partial):
  assert not p.args, "So far only keyword arguments are supported, here"
  fields = {k: v.default for k, v in inspect.signature(p.func).parameters.items()}
  fields = {k: v for k, v in fields.items() if v is not inspect.Parameter.empty}
  fields.update(p.keywords)
  nested = {k: partial_to_dict(partial(v)) for k, v in fields.items() if callable(v)}
  simple = {k: v for k, v in fields.items() if k not in nested}
  return {"__name__": p.func.__module__ + ":" + p.func.__qualname__, **simple, **nested}


def partial_from_dict(d: dict):
  d = {k: partial_from_dict(v) if isinstance(v, dict) and "__name__" in v else v for k, v in d.items()}
  module, name = d.pop("__name__").split(":")
  func = getattr(import_module(module), name)
  return partial(func, **d)
