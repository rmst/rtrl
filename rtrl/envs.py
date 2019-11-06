import atexit
import os
from dataclasses import dataclass, InitVar
import gym
from rtrl.wrappers import Float64ToFloat32, TimeLimitResetWrapper, NormalizeActionWrapper, RealTimeWrapper, \
  TupleObservationWrapper, AffineObservationWrapper, AffineRewardWrapper, PreviousActionWrapper
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
  # TODO: remove
  mean = np.array([-2.8978434e-01,  1.6660135e+00,  8.3246939e-02,  8.7178364e-02,
        2.4026206e-02,  6.5018900e-02,  1.4149654e-02,  1.0042821e-01,
       -6.2972151e-02, -5.7678702e-03,  3.1061701e-02,  2.9805478e-02,
       -6.7784175e-02,  4.7515810e-02, -5.1149428e-03,  5.3308241e-02,
        3.0382004e-04])
  std = np.array([0.23879883, 1.4998492 , 0.28086397, 0.29111406, 0.28239846,
       0.36456883, 0.3396931 , 0.28372395, 0.673492  , 0.7025841 ,
       1.4711759 , 5.872539  , 6.899377  , 7.830265  , 6.6277347 ,
       7.3030405 , 6.1115613 ])

  env = AffineObservationWrapper(env, -mean, 1/std)
  # env = AffineRewardWrapper(env, 0., 1.)
  return env


class Env(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.transition = (self.reset(), 0., True, {})

  def reset(self):
    return self.observation(self.env.reset())

  def step(self, action):
    next_state, reward, done, info = self.env.step(action)
    next_state = self.reset() if done else self.observation(next_state)
    self.transition = next_state, reward, done, info
    return self.transition

  def observation(self, observation):
    return observation


class GymEnv(Env):
  def __init__(self, seed_val=0, id: str = "Pendulum-v0", real_time: bool = False, normalize: bool = False):
    env = gym.make(id)
    if normalize:
      assert id.startswith("HalfCheetah")
      env = normalize_half_cheetah(env)
      # env = AffineObservationWrapper(env, 0., 0.1)

    env = Float64ToFloat32(env)
    env = TimeLimitResetWrapper(env)
    # env = DictObservationWrapper(env)
    assert isinstance(env.action_space, gym.spaces.Box)
    env = NormalizeActionWrapper(env)
    # env = DictActionWrapper(env)
    if real_time:
      env = RealTimeWrapper(env)
    else:
      env = TupleObservationWrapper(env)

    super().__init__(env)
    # self.seed(seed_val)


class AvenueEnv(Env):
  def __init__(self, seed_val=0, id: str = "RaceSolo-v0", real_time: bool = False):
    import avenue
    env = avenue.make(id.replace('-', '_'))
    # env = TimeLimitResetWrapper(env)
    # env = DictObservationWrapper(env)
    assert isinstance(env.action_space, gym.spaces.Box)
    env = NormalizeActionWrapper(env)
    # env = DictActionWrapper(env)
    if real_time:
      env = RealTimeWrapper(env)
    else:
      # Avenue environments are non-markovian. We don't want to give real-time methods an advantage by having the past action as part of it's state while non-real-time methods have not. I.e. we add the past action to the state below.
      env = PreviousActionWrapper(env)
      # env = TupleObservationWrapper(env)
    super().__init__(env)

    # bring images into right format: batch x channels x height x width
    (img_sp, vec_sp), *more = env.observation_space
    img_sp = gym.spaces.Box(img_sp.low.transpose(2, 0, 1), img_sp.high.transpose(2, 0, 1), dtype=img_sp.dtype)
    self.observation_space = gym.spaces.Tuple((gym.spaces.Tuple((img_sp, vec_sp)), *more))
    # self.seed(seed_val)

  def observation(self, observation):
    (img, vec), *more = observation
    return ((img.transpose(2, 0, 1), vec), *more)


def test_avenue():
  env = AvenueEnv(id="CityPedestrians-v0")
  env.reset()
  [env.step(env.action_space.sample()) for _ in range(1000)]
  (img, ), _, _, _ = env.step(env.action_space.sample())
  print('fjdk')