from rtrl.training import Training
from rtrl.util import partial

MjTest = partial(
  Training,
  epochs=3,
  rounds=5,
  steps=10,
  Agent=partial(memory_size=1000000),
  Env=partial(id="Pendulum-v0"),
)

MjTraining = partial(
  Training,
  epochs=50,
  rounds=20,
  steps=1000,
  Agent=partial(memory_size=1000000, batchsize=256),
  Env=partial(id='Walker2d-v2'),
)
