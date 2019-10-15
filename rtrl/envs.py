from dataclasses import dataclass, InitVar
import gym
from rtrl.wrappers import Float64ToFloat32, TimeLimitResetWrapper, NormalizeActionWrapper, RealTimeWrapper, \
  TupleObservationWrapper, AffineObservationWrapper, AffineRewardWrapper
import numpy as np


def mujoco_py_issue_424_workaround():
  """Mujoco_py generates files in site-packages for some reason.
  It causes trouble with docker and during runtime.
  https://github.com/openai/mujoco-py/issues/424
  """
  import os
  from os.path import dirname, join
  from shutil import rmtree
  import pkgutil
  path = join(dirname(pkgutil.get_loader("mujoco_py").path), "generated")
  [os.remove(join(path, name)) for name in os.listdir(path) if name.endswith("lock")]


def normalize_half_cheetah(env):
  mean = np.array((0.22799526154994965, 1.5053091049194336, 0.2822640538215637, 0.28506359457969666, 0.2727450430393219,
         0.38986486196517944, 0.33656004071235657, 0.28865376114845276, 0.7166818976402283, 0.761592447757721,
         1.6132901906967163, 5.744279384613037, 6.705165863037109, 7.640653610229492, 6.6329426765441895,
         7.241889476776123, 6.1558427810668945))
  std = np.array((0.051981840282678604, 2.2659554481506348, 0.07967299222946167, 0.08126124739646912,
         0.07438986003398895, 0.15199460089206696, 0.11327265202999115, 0.0833209902048111, 0.5136329531669617,
         0.5800230503082275, 2.602705240249634, 32.99674606323242, 44.959251403808594, 58.37958908081055,
         43.99592971801758, 52.44496536254883, 37.8943977355957))

  env = AffineObservationWrapper(env, -mean, 1/std)
  env = AffineRewardWrapper(env, 0., 1.)
  return env


@dataclass(eq=0)
class GymEnv(gym.Wrapper):
  seed_val: InitVar[int]  # the name seed is already taken by the gym.Env.seed function
  id: str = "Pendulum-v0"
  real_time: bool = False

  def __post_init__(self, seed_val):
    env = gym.make(self.id)
    if self.id.startswith("HalfCheetah"):
      env = normalize_half_cheetah(env)

    env = Float64ToFloat32(env)
    env = TimeLimitResetWrapper(env)
    # env = DictObservationWrapper(env)
    assert isinstance(env.action_space, gym.spaces.Box)
    env = NormalizeActionWrapper(env)
    # env = DictActionWrapper(env)
    if self.real_time:
      env = RealTimeWrapper(env)
    else:
      env = TupleObservationWrapper(env)

    super().__init__(env)
    self.seed(seed_val)


@dataclass(eq=0)
class AvenueEnv(gym.Wrapper):
  seed_val: InitVar[int]  # the name seed is already taken by the gym.Env.seed function
  id: str = "LaneFollowingTrack"
  real_time: bool = False

  def __post_init__(self, seed_val):
    import avenue
    env = avenue.make(self.id)
    env = TimeLimitResetWrapper(env)
    # env = DictObservationWrapper(env)
    assert isinstance(env.action_space, gym.spaces.Box)
    env = NormalizeActionWrapper(env)
    # env = DictActionWrapper(env)
    if self.real_time:
      env = RealTimeWrapper(env)
    super().__init__(env)
    self.seed(seed_val)
