from collections import deque
from copy import deepcopy, copy
from dataclasses import dataclass, InitVar
from functools import lru_cache, reduce
from itertools import chain

import rtrl.sac_models
import torch
from torch import optim
from torch.nn.functional import mse_loss

import rtrl.nn
from rtrl.memory import SimpleMemory, collate, partition
from rtrl.nn import PopArt, no_grad, copy_shared
from rtrl.util import shallow_copy, cached_property
import numpy as np


@dataclass(eq=0)
class Agent:
  observation_space: InitVar
  action_space: InitVar
  Model: type = rtrl.sac_models.Mlp
  OutputNorm: type = PopArt

  batchsize: int = 256  # training batch size
  memory_size: int = 1000000  # replay memory size
  lr: float = 0.0003
  discount: float = 0.99
  polyak: float = 0.995  # = 1 - 0.005
  keep_reset_transitions: int = 0
  reward_scale: float = 5.
  entropy_scale: float = 1.
  start_training: int = 10000
  device: str = None
  training_interval: int = 1
  model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

  num_updates = 0
  training_steps = 0

  def __post_init__(self, observation_space, action_space):
    device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = self.Model(observation_space, action_space)
    self.model: Agent.Model = model.to(device)
    self.model_target: Agent.Model = no_grad(deepcopy(self.model))

    self.policy_optimizer = optim.Adam(self.model.actor.parameters(), lr=self.lr)
    self.critic_optimizer = optim.Adam(chain.from_iterable(c.parameters() for c in self.model.critics), lr=self.lr)
    self.memory = SimpleMemory(self.memory_size, self.batchsize, device)

    self.output_norm = self.OutputNorm(dim=1).to(device)
    self.output_norm_target = self.OutputNorm(dim=1).to(device)

  def act(self, obs, r, done, info, train=False):
    stats = []
    action, _ = self.model.act(obs, r, done, info, train)

    if train:
      self.memory.append(np.float32(r), np.float32(done), info, obs, action)
      if len(self.memory) >= self.start_training and self.training_steps % self.training_interval == 0:
        stats += self.train(),
      self.training_steps += 1
    return action, stats

  def train(self):
    stats = {}

    obs, actions, rewards, next_obs, terminals = self.memory.sample()
    rewards, terminals = rewards[:, None], terminals[:, None]  # expand for correct broadcasting below

    new_action_distribution = self.model.actor(obs)
    new_actions = new_action_distribution.rsample()

    # actor loss
    new_actions_log_prob = new_action_distribution.log_prob(new_actions)[:, None]
    assert new_actions_log_prob.dim() == 2 and new_actions_log_prob.shape[1] == 1, "use Independent(Dist(...), 1) instead of Dist(...)"

    new_action_value = [c(obs, new_actions) for c in self.model.critics]
    new_action_value = reduce(torch.min, new_action_value)
    new_action_value = self.output_norm.unnormalize(new_action_value)

    loss_actor = self.entropy_scale * new_actions_log_prob.mean() - new_action_value.mean()
    loss_actor, = self.output_norm.normalize(loss_actor)
    stats.update(loss_actor=loss_actor.detach())

    # critic loss
    next_action_distribution = self.model_nograd.actor(next_obs)
    next_actions = next_action_distribution.sample()
    next_action_value = [c(next_obs, next_actions) for c in self.model_target.critics]
    next_action_value = reduce(torch.min, next_action_value)
    next_action_value = next_action_value - next_action_distribution.log_prob(next_actions)[:, None]
    next_action_value = self.output_norm_target.unnormalize(next_action_value)
    action_value_target = self.reward_scale * rewards + (1. - terminals) * self.discount * next_action_value

    self.output_norm.update(action_value_target)
    stats.update(v_mean=float(self.output_norm.m1), v_std=float(self.output_norm.std))
    [self.output_norm.update_lin(c[-1]) for c in self.model.critics]

    action_value_target = self.output_norm.normalize(action_value_target)

    action_values = [c(obs, actions) for c in self.model.critics]
    assert not action_value_target.requires_grad
    loss_critic = sum(mse_loss(av, action_value_target) for av in action_values)
    stats.update(loss_critic=loss_critic.detach())

    # update actor and critic
    self.critic_optimizer.zero_grad()
    loss_critic.backward()
    self.critic_optimizer.step()

    self.policy_optimizer.zero_grad()
    loss_actor.backward()
    self.policy_optimizer.step()

    # update target critics and normalizers
    with torch.no_grad():
      for t, n in zip(self.model_target.critics.parameters(), self.model.critics.parameters()):
        t.data += (1 - self.polyak) * (n - t)  # equivalent to t = α * t + (1-α) * n
    self.output_norm_target.m1 = self.polyak * self.output_norm_target.m1 + (1 - self.polyak) * self.output_norm.m1
    self.output_norm_target.std = self.polyak * self.output_norm_target.std + (1 - self.polyak) * self.output_norm.std

    self.num_updates += 1
    return dict(stats, memory_size=len(self.memory), updates=self.num_updates)


# === tests ============================================================================================================
def test_agent():
  from rtrl import partial, Training, run
  Sac_Test = partial(
    Training,
    epochs=3,
    rounds=5,
    steps=100,
    Agent=partial(Agent, memory_size=1000000, start_training=256, batchsize=4),
    Env=partial(id="Pendulum-v0", real_time=0),
  )
  run(Sac_Test)


def test_agent_avenue():
  from rtrl import partial, Training, run
  from rtrl.envs import AvenueEnv
  from rtrl.sac_models import ConvModel
  Sac_Avenue_Test = partial(
    Training,
    epochs=3,
    rounds=5,
    steps=300,
    Agent=partial(
      Agent, device='cpu', training_interval=4, lr=0.0001, memory_size=200000,
      start_training=10000, batchsize=100, Model=partial(
        ConvModel)),
    # Env=partial(id="Pendulum-v0", real_time=True),
    Env=partial(AvenueEnv, real_time=0),
  )
  run(Sac_Avenue_Test)


if __name__ == "__main__":
  test_agent()
  # test_agent_avenue()
