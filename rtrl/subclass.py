from copy import deepcopy
from functools import partial, partialmethod
from importlib import import_module
from typing import Union
from uuid import uuid4


def configuration(cls: type):
  """this defines what subset of a class' attributes we consider to be the configuration"""
  return {k: getattr(cls, k) for k, t in getattr(cls, "__annotations__", {}).items() if hasattr(cls, k)}  # all annotated attributes as dict


def conf_to_dict(cls: type):
  config_cls, base = cls.__bases__
  assert config_cls is Conf
  attrs = {k: conf_to_dict(v) if isinstance(v, type) else v for k, v in configuration(cls).items()}
  return dict(attrs, __class__=base.__module__ + "." + base.__qualname__)


def conf_from_dict(d: dict):
  *module, name = d["__class__"].split(".")
  base = getattr(import_module(".".join(module)), name)
  return conf(base, **{k: conf_from_dict(v) if isinstance(v, dict) else v for k, v in d.items() if k != "__class__"})


class Conf:
  def __setattr__(self, k, v):
    if k in configuration(self.__class__):
      raise AttributeError(f"Attribute '{k}' is read only")
    return super().__setattr__(k, v)

  def __reduce__(self):
    state = self.__getstate__() if hasattr(self, "__getstate__") else vars(self)
    config_cls, base = self.__class__.__bases__
    assert config_cls is Conf
    return _reconstruct, (conf_to_dict(self.__class__),), state


def conf(__class__: type = None, **kwargs):
  if __class__ is None:
    return partial(conf, **kwargs)
  attrs = configuration(__class__)
  if issubclass(__class__, Conf):
    config_cls, __class__ = __class__.__bases__
    assert config_cls is Conf
  diff = kwargs.keys() - attrs.keys()
  assert not diff, f"Only existing attributes can be overridden, not {diff}"
  kwargs = {k: v(attrs[k]) if isinstance(v, partial) and v.func==conf else v for k, v in attrs.items()}  # insert missing default classes into partial confs
  attrs = dict(attrs, **kwargs)
  attrs = {k: conf(v) if isinstance(v, type) else v for k, v in attrs.items()}  # replace non-conf types
  new_cls = type(f"conf({__class__.__name__}, ...)", (Conf, __class__), attrs)
  return new_cls


def _reconstruct(d):
  cls = conf_from_dict(d)
  return cls.__new__(cls)


if __name__ == "__main__":
  class A:
    b: int = None

  B = conf(A, b=3)
  b = B()
  b.c = 9
  c = deepcopy(b)
  assert vars(b) == vars(c)
  assert vars(b.__class__) == vars(c.__class__)
  print("success")
