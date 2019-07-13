from copy import deepcopy
from itertools import chain

import rtrl.models
import torch
from torch import optim
from torch.nn.functional import mse_loss

from rtrl.memory import SimpleMemory, collate, partition
from rtrl.nn import PopArt, no_grad, copy_shared
from rtrl.util import apply_kwargs


class Agent:
  M = rtrl.models.Mlp
  OutputNorm = PopArt

  batchsize: int = 256  # training batch size
  memory_size: int = 1000000  # replay memory size
  lr_actor: float = 0.0003
  lr: float = 0.0003
  discount: float = 0.99
  polyak: float = 0.995  # = 1 - 0.005
  device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
  policy_freq: int = 1
  target_freq: int = 1
  keep_reset_transitions: int = 0
  reward_scale: float = 5.
  entropy_scale: float = 1.  # called alpha in RLKit
  start_training: int = 1000

  def __init__(self, obsp, acsp, **kwargs):
    apply_kwargs(self, kwargs)

    model = self.M(obsp, acsp)
    self.model = model.to(self.device)
    self.model_target: Agent.M = no_grad(deepcopy(self.model))
    self.model_nograd: Agent.M = no_grad(copy_shared(self.model))

    self.policy_optimizer = optim.Adam(self.model.actor.parameters(), lr=self.lr_actor)
    self.critic_optimizer = optim.Adam(chain(self.model.value.parameters(), *(c.parameters() for c in self.model.critics)), lr=self.lr)
    self.memory = SimpleMemory(self.memory_size, self.batchsize, self.device)

    self.num_updates = 0
    self.is_training = False

    self.outnorm = self.OutputNorm(1, device=self.device)
    self.outnorm_target = self.OutputNorm(1, device=self.device)

  def act(self, r, done, info, obs, train=False):
    stats = {}
    obs_col = collate((obs,))
    action_distribution = self.model_nograd.actor(obs_col)
    action_col = action_distribution.sample()
    action, = partition(action_col)

    if train:
      self.memory.append(r, float(done), info, obs, action)

      if not self.is_training and len(self.memory) > self.start_training:
        print(f"Starting training with memory size {len(self.memory)}")
        self.is_training = True

      if self.is_training:
        stats.update(self.train())

    return action, stats

  def train(self):
    stats = {}

    obs, actions, rewards, next_obs, terminals = self.memory.sample()
    rewards, terminals = rewards[:, None], terminals[:, None]  # expand for correct broadcasting below

    v_pred = self.model.value(obs)

    policy_outputs = self.model.actor(obs)  # should include logprob
    assert isinstance(policy_outputs.base_dist, rtrl.models.TanhNormal)
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
    policy_loss = self.outnorm.normalize(policy_loss)

    self.policy_optimizer.zero_grad()
    policy_loss.backward()
    self.policy_optimizer.step()

    stats.update(loss_actor=policy_loss.detach())

    if self.num_updates % self.target_freq == 0:
      with torch.no_grad():
        for t, n in zip(self.model_target.parameters(), self.model.parameters()):
          t.data += (1 - self.polyak) * (n - t)  # equivalent to t = α * t + (1-α) * n
      self.outnorm_target.m1 = self.polyak * self.outnorm_target.m1 + (1-self.polyak) * self.outnorm.m1
      self.outnorm_target.std = self.polyak * self.outnorm_target.std + (1 - self.polyak) * self.outnorm.std

    self.num_updates += 1

    stats.update(memory_size=len(self.memory), updates=self.num_updates)
    return stats

  def __getstate__(self):
    return {k: v for k, v in vars(self).items() if k not in (
      "memory",
      "compute_reward",
      "logger",
      "opt_actor",
      "opt_critic",
      "model_target",
      "model_nograd",
    )}

  def __setstate__(self, state):
    self.__dict__.update(state)
    assert not hasattr(self, "model_nograd")
    self.model_nograd: Agent.M = no_grad(copy_shared(self.model))
