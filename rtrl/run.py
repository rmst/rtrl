from copy import deepcopy, copy
from time import time

import gym

import rtrl.sac
from rtrl.serialization import Directory
from rtrl.util import apply_kwargs
from rtrl.wrappers import TimeLimitResetWrapper, DictObservationWrapper, NormalizeActionWrapper, DictActionWrapper, \
  Float64ToFloat32
import torch.multiprocessing as mp


class GymEnv(gym.Wrapper):
  id: str = "Pendulum-v0"

  def __init__(self, **kwargs):
    apply_kwargs(self, kwargs)

    env = gym.make(self.id)
    env = Float64ToFloat32(env)
    env = TimeLimitResetWrapper(env)
    env = DictObservationWrapper(env)
    assert isinstance(env.action_space, gym.spaces.Box)
    env = NormalizeActionWrapper(env)
    env = DictActionWrapper(env)

    super().__init__(env)


class Test:
  steps: int = 10000

  def __init__(self, agent: rtrl.sac.Agent, Env=None, seed=None):
    self.agent = agent
    self.Env = Env
    self.seed = seed

  def run(self):
    env = self.Env(seed=self.seed)
    r, done, info = 0., True, {}
    for timestep in range(self.steps):
      if done:
        obs = env.reset()
      action, stats = self.agent.act(r, done, info, obs, train=False)
      obs, r, done, info = env.step(action)
      # logger.log(**{'test_' + k: v for k, v in test_env.stats().items()})


class Training:
  Env = GymEnv
  Agent = rtrl.sac.Agent
  Test = Test
  steps: int = 20000  # number of timesteps to run
  teststep: int = 100  # timesteps between tests
  seed: int = 0
  epochs: int = 50

  def __init__(self):
    # train_env = gyn.wrappers.StatsWrapper(train_env, stats_window=5 * self.epoch_length * self.logstep)
    self.env = self.Env(seed=self.seed)
    print(self.env)
    # test_env = self.Env(seed=self.seed + 1000)
    # test_env = gyn.wrappers.StatsWrapper(test_env, stats_window=5 * self.epoch_length * self.teststep)

    self.agent = self.Agent(self.env.observation_space, self.env.action_space)
    # args_help(kwargs)
    self.stats = {}
    self.epoch = 0

  def __getfiles__(self):
    state = copy(self)
    del state.env  # not saving that for now
    del state.agent, state.stats
    return dict(state=vars(state), agent=self.agent, stats=Directory(self.stats))

  def __setfiles__(self, files):
    self.agent = files.pop("agent")
    vars(self).update(files.pop("state"))
    self.env = self.Env(seed=self.seed + self.epoch)

  def run_epoch(self):
    print(f"start epoch {self.epoch}")
    pool = mp.Pool(1)
    results = []

    r, done, info = 0., True, {}
    for step in range(self.steps):
      if self.teststep and step % self.teststep == 0:
        print("Start test at step", step)
        test = Test(deepcopy(self.agent), self.Env, self.seed+self.epochs+self.epoch+step)
        results += (dict(time=time(), step=self.epoch*self.steps+step), pool.apply_async(test.run)),

      if done:
        obs = self.env.reset()

      action, stats = self.agent.act(r, done, info, obs, train=True)
      obs, r, done, info = self.env.step(action)

    self.stats = {k: v for k, v in self.stats.items()}
    pool.close()
    self.epoch += 1

  def __del__(self):
    self.env.close()
