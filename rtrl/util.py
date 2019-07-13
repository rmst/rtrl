

def apply_kwargs(obj, kwargs):
  # TODO: rename to setattributes
  for k, v in kwargs.items():
    assert hasattr(obj.__class__, k), f"Can't set {repr(k)} on {obj}"
    setattr(obj, k, v)
