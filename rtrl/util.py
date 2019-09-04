import functools
import inspect
from dataclasses import is_dataclass, dataclass, make_dataclass, fields, Field
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


# === cached properties ================================================================================================
def cached_property(fun):
  return property(partial(get_cached, fun, WeakKeyDictionary()))


def get_cached(fun, cache, obj):
  c = cache.get(obj, None)
  if c is None:
    c = cache[obj] = fun()
  return c


# === misc =============================================================================================================
class confclass:
  def __new__(cls, other: Type[T], **kwargs) -> Union["confclass", Type[T]]:
    obj = super().__new__(cls)
    obj._cls = other
    obj._kwargs = kwargs
    return obj
  # def __init__(self, cls: Type[T], **kwargs):
  #   self.cls = cls
  #   self.kwargs = kwargs

  def __call__(self, *args, **kwargs) -> Union[T, Type[T], Callable[[Any], Type[T]]]:
    kwargs = dict(self._kwargs, **kwargs)
    if ... in args:
      assert len(args) == 1
      return confclass(self._cls, **kwargs)
    else:
      return self._cls(*args, **kwargs)


confclass(pd.Series)



def test(t: Type[T]) -> Union[T, Type[T]]:
  return t

class B(property):
  pass

@dataclass
class A:
  a: int = 3

  @property
  @functools.lru_cache()
  def c(self):
    return 3

  @c.setter
  def c(self, c):
    pass
  
  def __instancecheck__(self, instance):
    self.b = 3
    a = self.c
A(a=4)


class B:
  def __init__(self, a):
    pass


B(a=4)


def apply_kwargs(obj, kwargs):
  # TODO: rename to setattributes
  for k, v in kwargs.items():
    assert hasattr(obj.__class__, k), f"Can't set {repr(k)} on {obj}"
    setattr(obj, k, v)


def default():
  raise ValueError("This is a dummy function not meant to be called. It can be used within a nested `partial` where it is replaced with the default value of a keyword argument.")


class partial(functools.partial):
  """Like `functools.partial`. However, when used as a keyword argument within another `partial` and if `default` is supplied as the function, `default` will be replaced with the default value for that keyword argument."""
  def __new__(cls, func: Type[T], *args, **kwargs) -> Union[T, Type[T]]:
    for k, v in kwargs.items():
      if isinstance(v, partial) and v.func is default:
        kwargs[k] = partial(inspect.signature(func).parameters[k].default, *v.args, **v.keywords)
    return super().__new__(cls, func, *args, **kwargs)

  def __eq__(self, other: functools.partial):
    return all((self.func == other.func,
                self.args == other.args,
                self.keywords == other.keywords))

  def __getattr__(self, item):
    return self.keywords[item] if item in self.keywords else getattr(self.func, item)

  def to_string(self, prefix="") -> str:
    assert not self.args, "So far only keyword arguments are supported, here"
    fields = {k.name: v.default for k, v in inspect.signature(self.func).parameters.items()}
    fields.update(self.keywords)
    fields = dict(chain.from_iterable(v.to_string(k + ".") if isinstance(v, partial) else [(k, v)] for k, v in fields.items()))


partial(pd.Series, a=3)()