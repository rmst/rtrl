from rtrl.training import Training
from rtrl.util import partial

MjTest = partial(
  Training,
  epochs=100,
  rounds=100,
  steps=10,
  Agent=partial(memory_size=1000000),
  Env=partial(id="Pendulum-v0"),
)

MjTraining = partial(
  Training,
  epochs=100,
  rounds=100,
  steps=100,
  Agent=partial(memory_size=1000000, batchsize=256),
  Env=partial(id='Walker2d-v2'),
)
