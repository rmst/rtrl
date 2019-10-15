from copy import deepcopy
from dataclasses import dataclass

import torch
from torch.nn.functional import mse_loss

import rtrl.sac
from rtrl.memory import SimpleMemory
from rtrl.nn import no_grad, copy_shared
import rtrl.models


@dataclass(eq=0)
class Agent(rtrl.sac.Agent):
  Model: type = rtrl.models.MlpRTDouble
  loss_alpha: float = 0.2

  def __post_init__(self, observation_space, action_space):
    device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = self.Model(observation_space, action_space)
    self.model = model.to(device)
    self.model_target = no_grad(deepcopy(self.model))
    # polyak(self.model_target, self.model, 0)  # ensure they have equal parameter values

    self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    self.memory = SimpleMemory(self.memory_size, self.batchsize, device)

    self.num_updates = 0
    self.is_training = False

    self.outnorm = self.OutputNorm(1).to(device)
    self.outnorm_target = self.OutputNorm(1).to(device)

  def train(self):
    stats = {}

    # indices = np.random.randint(0, len(self.memory)-1, self.batchsize)
    obs, actions, rewards, next_obs, terminals = self.memory.sample()
    rewards, terminals = rewards[:, None], terminals[:, None]  # expand for correct broadcasting below

    action_distribution, v_preds = self.model(obs)
    # assert isinstance(policy_outputs.base_dist, (TanhNormal, agents.sac.models.ConcatDirac))
    new_actions = action_distribution.rsample()
    log_pi = action_distribution.log_prob(new_actions)[:, None]
    # policy_mean = policy_outputs.base_dist.normal_mean
    # policy_log_std = torch.log(policy_outputs.base_dist.normal_std)
    assert log_pi.dim() == 2 and log_pi.shape[1] == 1, "use Independent(Normal(...), 1) instead of Normal(...)"

    # no alpha loss / adaptive entropy scaling
    new_next_obs = (next_obs[0], new_actions)
    new_next_obs_tgt = (next_obs[0], new_actions.detach())
    _, v2s_target = self.model_target(new_next_obs_tgt)
    _, v2s = self.model_nograd(new_next_obs)

    v2_target, _ = torch.stack(v2s_target, 2).min(2)
    v2, _ = torch.stack(v2s, 2).min(2)

    v_target = self.reward_scale * rewards
    v_target -= self.entropy_scale * log_pi.detach()
    v_target += (1. - terminals) * self.discount * self.outnorm_target.unnormalize(v2_target)

    self.outnorm.update(v_target)
    stats.update(v_mean=float(self.outnorm.m1), v_std=float(self.outnorm.std))

    v_target = self.outnorm.normalize(v_target)
    [self.outnorm.update_lin(x) for x in self.model.v_out]

    # assert not v_target.requires_grad
    v_loss = sum(mse_loss(v_pred, v_target) for v_pred in v_preds)
    # v_loss = fu.mse_loss(v_pred, v_target)

    # Policy Loss (with reparameterization)
    # policy_loss = self.entropy_scale * log_pi.mean() - self.outnorm.unnormalize(v2.mean())
    policy_loss = self.entropy_scale * log_pi.mean() - ((1. - terminals) * self.discount * self.outnorm.unnormalize(v2)).mean()
    policy_loss = self.outnorm.normalize(policy_loss)

    # Update Networks
    self.opt.zero_grad()
    # alpha = self.v_loss / (self.v_loss + 1)
    loss_total = self.loss_alpha * policy_loss + (1-self.loss_alpha) * v_loss
    loss_total.backward()
    self.opt.step()
    stats.update(loss_total=loss_total.detach())
    stats.update(loss_value=v_loss.detach())
    stats.update(loss_actor=policy_loss.detach())

    with torch.no_grad():
      for t, n in zip(self.model_target.parameters(), self.model.parameters()):
        t.data += (1 - self.polyak) * (n - t)  # equivalent to t = α * t + (1-α) * n
    self.outnorm_target.m1 = self.polyak * self.outnorm_target.m1 + (1-self.polyak) * self.outnorm.m1
    self.outnorm_target.std = self.polyak * self.outnorm_target.std + (1 - self.polyak) * self.outnorm.std

    self.num_updates += 1

    stats.update(memory_size=len(self.memory), updates=self.num_updates, entropy_scale=self.entropy_scale)
    return stats


if __name__ == "__main__":
  from rtrl import partial, Training, run

  Rtac_Test = partial(
    Training,
    epochs=3,
    rounds=5,
    steps=100,
    Agent=partial(Agent, memory_size=1000000, start_training=256, batchsize=4),
    Env=partial(id="Pendulum-v0", real_time=True),
  )
  run(Rtac_Test)
