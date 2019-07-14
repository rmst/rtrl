

def apply_kwargs(obj, kwargs):
  # TODO: rename to setattributes
  for k, v in kwargs.items():
    assert hasattr(obj.__class__, k), f"Can't set {repr(k)} on {obj}"
    setattr(obj, k, v)


def print_dict(d):
  print('-' * 75)
  for n, v in d.items():
    print("  " + n + " = " + repr(v))
  print('-' * 75)