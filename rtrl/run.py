import datetime
import os
import sys
from copy import deepcopy
from functools import partial
from random import randrange, random

import gym
import torch

import rtrl
import rtrl.sac
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
  Env = GymEnv
  steps = 10000
  seed: int = randrange(1 << 32)  # could also be provided via $RANDOM
  agent_path: str = 'agent'

  def __init__(self, agent: rtrl.sac.Agent = None, **kwargs):
    apply_kwargs(self, kwargs)

    env = self.Env(seed=self.seed)

    # with open(self.agent_path) as f:
    #   agent: rtrl.sac.Agent = torch.load(f)

    r, done, info = 0., True, {}
    for timestep in range(self.steps):
      if done:
        obs = env.reset()

      action, stats = agent.act(r, done, info, obs, train=False)
      obs, r, done, info = env.step(action)


      # logger.log(**{'test_' + k: v for k, v in test_env.stats().items()})


class Train:
  Env = GymEnv
  Agent = rtrl.sac.Agent
  steps: int = 20000  # number of timesteps to run
  # logger: str = ''  # TODO: make class based
  # model_path: str = ''
  logstep: int = 1
  savestep: int = 100  # timesteps between model checkpoints
  teststep: int = 100  # timesteps between tests
  seed: int = randrange(1 << 32)  # could also be provided via $RANDOM

  def __init__(self, **kwargs):
    apply_kwargs(self, kwargs)

    # serializable_args.update(agents.logger.git_info(os.path.dirname(__file__)))
    # serializable_args.update(datetime=datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'))
    # serializable_args.update({k: os.environ.get(k, '') for k in os.environ.get('AGENTS_LOG_ENV', '').strip().split()})
    # logger: agents.logger.Logger = agents.logger.make(self.logger, serializable_args, code=__name__)

    env = self.Env(seed=self.seed)
    # train_env = gyn.wrappers.StatsWrapper(train_env, stats_window=5 * self.epoch_length * self.logstep)

    print(env)
    test_proc: mp.Process = None
    # test_env = self.Env(seed=self.seed + 1000)
    # test_env = gyn.wrappers.StatsWrapper(test_env, stats_window=5 * self.epoch_length * self.teststep)

    agent = self.Agent(env.observation_space, env.action_space)
    # args_help(kwargs)

    r, done, info = 0., True, {}
    for step in range(self.steps):
      # print("start epoch {} after {} steps".format(epoch, env.total_steps))
      # logger.log(cur_epoch=epoch)

      if self.teststep and step % self.teststep == 0:
        print("Start test at step", step)
        if test_proc is not None:
          test_proc.join()
        test_agent = deepcopy(agent)
        # test_agent.share_memory()
        test_proc = mp.Process(target=Test, kwargs=dict(agent=test_agent, Env=self.Env, seed=self.seed+1000))
        test_proc.start()
      # if self.model_path and epoch % self.savestep == 0:
      #   print('model saved')
      #   storage.save(argformat(self.model_path, {}), dumps_torch(agent))

      if done:
        obs = env.reset()

      action, stats = agent.act(r, done, info, obs, train=True)
      obs, r, done, info = env.step(action)

      # train_stats = agent.train(train_actor.experience())
      # logger.append(**train_stats)
      #
      # if epoch % self.logstep == 0:
      #   logger.log(**train_env.stats())
      #   logger.flush()

    env.close()
    test_proc.join()
