from copy import deepcopy
from functools import partial, partialmethod
from uuid import uuid4


def hp__setattr__(__setattr__, self, k, v):
  if k in hyperparameters(self.__class__):
    raise AttributeError(f"Attribute '{k}' is read only")
  return __setattr__(self, k, v)


def hp__reduce__(self):
  base, = self.__class__.__bases__  # self.__class__ has a single base class (we created it with subclass)
  state = self.__getstate__() if hasattr(self, "__getstate__") else vars(self)
  return reconstruct, (base, hyperparameters(self.__class__)), state


def hyperparameters(cls: type):
  return {k: v for k, v in vars(cls).items() if k in getattr(cls, "__annotations__", {})}


def hyperparameterize(cls: type):
  cls.__setattr__ = lambda self, k, v, f=cls.__setattr__: hp__setattr__(f, self, k, v)
  cls.__reduce__ = hp__reduce__
  return sub(type("dummy", (cls,), hyperparameters(cls)))


def reconstruct(base, params):
  cls = sub(type("dummy", (base,), params))
  return cls.__new__(cls)


def sub(cls, **kwargs):
  base, = cls.__bases__
  kwargs = dict(hyperparameters(cls), **kwargs)
  key = (base, frozenset(kwargs.items()))
  if key not in class_dict:
    class_dict[key] = type(base.__name__ + f"_{hash(key)}", (base,), kwargs)
  return class_dict[key]


class_dict = {}


def subclass(cls, **kwargs):
  name =
  diff = kwargs.keys() - vars(cls).keys()
  assert not diff, f"Only existing attributes can be overridden, not {diff}"
  kwargs.update(__reduce__=reduce, __is_partial_subclass__=True)
  return type(name, (cls,), kwargs)


def subclass_and_instantiate(cls, attributes):
  sub = subclass(cls, **attributes)
  return sub.__new__(sub)


def reduce(self):
  base, = self.__class__.__bases__  # self.__class__ has a single base class (we created it with subclass)
  attributes = {k: v for k, v in vars(self.__class__).items() if not k.startswith("__")}
  state = self.__getstate__() if hasattr(self, "__getstate__") else vars(self)
  return subclass_and_instantiate, (base, attributes), state


if __name__ == "__main__":
  class A:
    b = None

  B = subclass(A, b=3)
  b = B()
  b.c = 9
  c = deepcopy(b)
  assert vars(b) == vars(c)
  assert vars(b.__class__) == vars(c.__class__)
  print("success")
