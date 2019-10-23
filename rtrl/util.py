import functools
import inspect
import io
import json
import os
import pickle
import subprocess
import weakref
from dataclasses import is_dataclass, dataclass, make_dataclass, fields, Field
from importlib import import_module
from itertools import chain
from typing import TypeVar, Union, Type, Callable, Any, Dict
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
   Then, the outer `partial` will insert the appropriate default value as the function. """

  if func is not default:
    for k, v in kwargs.items():
      if isinstance(v, functools.partial) and v.func is default:
        kwargs[k] = partial(inspect.signature(func).parameters[k].default, *v.args, **v.keywords)
  return functools.partial(func, *args, **kwargs)


FKEY = '+'


def partial_to_dict(p: functools.partial, version="3"):
  assert not p.args, "So far only keyword arguments are supported, here"
  fields = {k: v.default for k, v in inspect.signature(p.func).parameters.items()}
  fields = {k: v for k, v in fields.items() if v is not inspect.Parameter.empty}
  diff = p.keywords.keys() - fields.keys()
  assert not diff, f"There are invalid keywords present: {diff}"
  fields.update(p.keywords)
  nested = {k: partial_to_dict(partial(v), version="") for k, v in fields.items() if callable(v)}
  simple = {k: v for k, v in fields.items() if k not in nested}
  output = {FKEY: p.func.__module__ + ":" + p.func.__qualname__, **simple, **nested}
  return dict(output, __format_version__=version) if version else output


def partial_from_dict(d: dict):
  d = d.copy()
  assert d.pop("__format_version__", "3") == "3"
  d = {k: partial_from_dict(v) if isinstance(v, dict) and FKEY in v else v for k, v in d.items()}
  func = get_class_or_function(d.pop(FKEY) or "rtrl.util:default")
  return partial(func, **d)


def get_class_or_function(func):
  module, name = func.split(":")
  return getattr(import_module(module), name)


def partial_from_args(func: Union[str, callable], kwargs: Dict[str, str]):
  # print(func, kwargs)  # useful to visualize the parsing process
  func = get_class_or_function(func) if isinstance(func, str) else func
  keys = {k.split('.')[0] for k in kwargs}
  keywords = {}
  for key in keys:
    param = inspect.signature(func).parameters[key]
    value = kwargs.get(key, param.default)
    if param.annotation is type:
      sub_keywords = {k.split('.', 1)[1]: v for k, v in kwargs.items() if k.startswith(key + '.')}
      keywords[key] = partial_from_args(value, sub_keywords)
    else:
      keywords[key] = param.annotation(value)
  return partial(func, **keywords)


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


# === git ==============================================================================================================
def get_output(*args, default='', **kwargs):
  try:
    output = subprocess.check_output(*args, universal_newlines=True, **kwargs)
    return output.rstrip("\n")  # skip trailing newlines as done in bash
  except subprocess.CalledProcessError:
    return default


def git_info(path=None):
  """returns a dict with information about the git repo at path (path can be a sub-directory of the git repo)
  """
  import __main__
  path = path or os.path.dirname(__main__.__file__)
  rev = get_output('git rev-parse HEAD'.split(), cwd=path)
  count = int(get_output('git rev-list HEAD --count'.split(), cwd=path))
  status = get_output('git status --short'.split(), cwd=path)  # shows uncommited modified files
  commit_date = get_output("git show --quiet --date=format-local:%Y-%m-%dT%H:%M:%SZ --format=%cd".split(), cwd=path, env=dict(TZ='UTC'))
  desc = get_output(['git', 'describe', '--long', '--tags', '--dirty', '--always', '--match', r'v[0-9]*\.[0-9]*'], cwd=path)
  message = desc + " " + ' '.join(get_output(['git', 'log', '--oneline', '--format=%B', '-n', '1', "HEAD"], cwd=path).splitlines())

  url = get_output('git config --get remote.origin.url'.split(), cwd=path).strip()
  # if on github, change remote to a meaningful https url
  if url.startswith('git@github.com:'):
    url = 'https://github.com/' + url[len('git@github.com:'):-len('.git')] + '/commit/' + rev
  elif url.startswith('https://github.com'):
    url = url[:len('.git')] + '/commit/' + rev

  return dict(url=url, rev=rev, count=count, status=status, desc=desc, date=commit_date, message=message)
