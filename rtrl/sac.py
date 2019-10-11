from collections import deque
from copy import deepcopy, copy
from dataclasses import dataclass, InitVar
from functools import lru_cache
from itertools import chain

import rtrl.models
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
  Model: type = rtrl.models.Mlp
  OutputNorm: type = PopArt

  batchsize: int = 256  # training batch size
  memory_size: int = 1000000  # replay memory size
  lr: float = 0.0003
  discount: float = 0.99
  polyak: float = 0.995  # = 1 - 0.005
  keep_reset_transitions: int = 0
  reward_scale: float = 5.
  entropy_scale: float = 1.
  start_training: int = 10000  # 1000
  device: str = "cuda"

  model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

  def __post_init__(self, observation_space, action_space):
    self.device = self.device if "cuda" in self.device and torch.cuda.is_available() else "cpu"
    model = self.Model(observation_space, action_space)
    self.model: Agent.Model = model.to(self.device)
    self.model_target: Agent.Model = no_grad(deepcopy(self.model))

    self.policy_optimizer = optim.Adam(self.model.actor.parameters(), lr=self.lr)
    self.critic_optimizer = optim.Adam(chain(self.model.value.parameters(), *(c.parameters() for c in self.model.critics)), lr=self.lr)
    self.memory = SimpleMemory(self.memory_size, self.batchsize, self.device)

    self.num_updates = 0

    self.outnorm = self.OutputNorm(dim=1).to(self.device)
    self.outnorm_target = self.OutputNorm(dim=1).to(self.device)

  def act(self, obs, r, done, info, train=False):
    stats = {}
    action, _ = self.model.act(obs, r, done, info)

    if train:
      self.memory.append(np.float32(r), np.float32(done), info, obs, action)
      if len(self.memory) >= self.start_training:
        stats.update(self.train())

    return action, stats

  def train(self):
    stats = {}

    obs, actions, rewards, next_obs, terminals = self.memory.sample()
    rewards, terminals = rewards[:, None], terminals[:, None]  # expand for correct broadcasting below

    v_pred = self.model.value(obs)

    policy_outputs = self.model.actor(obs)  # should include logprob
    assert isinstance(policy_outputs.base_dist, rtrl.nn.TanhNormal)
    new_actions = policy_outputs.rsample()
    log_pi = policy_outputs.log_prob(new_actions)[:, None]
    assert log_pi.dim() == 2 and log_pi.shape[1] == 1, "use Independent(Normal(...), 1) instead of Normal(...)"

    # QF Loss
    target_v_values = self.model_target.value(next_obs)
    q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * self.outnorm_target.unnormalize(target_v_values)

    self.outnorm.update(q_target)
    stats.update(v_mean=float(self.outnorm.m1), v_std=float(self.outnorm.std))

    q_target = self.outnorm.normalize(q_target)
    [self.outnorm.update_lin(c[-1]) for c in self.model.critics]
    self.outnorm.update_lin(self.model.value[-1])

    q_preds = [c(obs, actions) for c in self.model.critics]
    assert not q_target.requires_grad
    qf_loss = sum(mse_loss(q_pred, q_target) for q_pred in q_preds)

    stats.update(loss_critic=qf_loss.detach())

    # VF Loss
    q_new_actions_multi = [c(obs, new_actions) for c in self.model.critics]
    q_new_actions, _ = torch.stack(q_new_actions_multi, 2).min(2)

    v_target = self.outnorm.unnormalize(q_new_actions) - self.entropy_scale * log_pi
    # assert not v_target.requires_grad
    vf_loss = mse_loss(v_pred, self.outnorm.normalize(v_target.detach()))

    # Update Networks
    self.critic_optimizer.zero_grad()
    qf_loss.backward()
    vf_loss.backward()
    self.critic_optimizer.step()
    stats.update(loss_value=vf_loss.detach())

    policy_loss = self.entropy_scale * log_pi.mean() - self.outnorm.unnormalize(q_new_actions.mean())
    policy_loss, = self.outnorm.normalize(policy_loss)

    self.policy_optimizer.zero_grad()
    policy_loss.backward()
    self.policy_optimizer.step()

    stats.update(loss_actor=policy_loss.detach())

    with torch.no_grad():
      for t, n in zip(self.model_target.parameters(), self.model.parameters()):
        t.data += (1 - self.polyak) * (n - t)  # equivalent to t = α * t + (1-α) * n
    self.outnorm_target.m1 = self.polyak * self.outnorm_target.m1 + (1-self.polyak) * self.outnorm.m1
    self.outnorm_target.std = self.polyak * self.outnorm_target.std + (1 - self.polyak) * self.outnorm.std

    self.num_updates += 1
    return dict(stats, memory_size=len(self.memory), updates=self.num_updates)


if __name__ == "__main__":
  from rtrl import partial, Training, run

  Sac_Test = partial(
    Training,
    epochs=3,
    rounds=5,
    steps=100,
    Agent=partial(Agent, memory_size=1000000, start_training=256),
    Env=partial(id="Pendulum-v0"),
  )
  run(Sac_Test)
