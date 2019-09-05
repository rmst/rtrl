"""
This doesn't work yet. It is just a copy of what was in the old repo.

"""

import rtrl


class Agent(rtrl.sac.Agent):
  Model = rtrl.models.MlpRTDouble

  rec_dim: int = 0
  # v_loss: float = 100.

  def actor(self, x, train=False):
    # x = gyn.wrappers.ToPytorch(x)
    x = gyn.wrappers.SelectActionComponent(x, key="a")
    x = gyn.wrappers.Transpose(x, device=self.device)
    x = ActionSlice(x, dim=self.rec_dim)
    x = ActionDelay(x, device=self.device)
    x = gyn.wrappers.TransitionBuffer(x, keep_reset_transitions=self.keep_reset_transitions)
    x = self.A(x, self, train=train)
    return x

  # noinspection PyMissingConstructor
  def __init__(self, obsp, acsp, rec_dim=None, **kwargs):
    apply_kwargs(self, kwargs)

    if rec_dim is not None: self.rec_dim = rec_dim
    acsp = deepcopy(acsp)
    acsp.shape = (acsp.shape[0] + self.rec_dim, )

    obsp = deepcopy(obsp)
    obsp.spaces["s"].shape = (obsp.spaces["s"].shape[0] + acsp.shape[0],)

    model = self.M(obsp, acsp)
    self.model = model.to(self.device)
    self.model_target: self.M = no_grad(deepcopy(self.model))
    # polyak(self.model_target, self.model, 0)  # ensure they have equal parameter values

    self.model_nograd: self.M = no_grad(copy_shared(self.model))
    self.model_eval: self.M = no_grad(copy_shared(self.model))
    self.model_eval.train(False)

    self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
    self.memory = SimpleMemory(self.memory_size, self.batchsize, self.device)

    self.num_updates = 0
    self.is_training = False

    self.outnorm = self.OutputNorm(1, device=self.device)
    self.outnorm_target = self.OutputNorm(1, device=self.device)

    # ienv = gyn.Interface(obsp, acsp, self.batchsize)
    # self.ienv_train =

  def train_single(self):
    stats = {}

    # indices = np.random.randint(0, len(self.memory)-1, self.batchsize)
    obs, actions, rewards, next_obs, terminals, rm_stats = self.memory.sample()
    obs: STATE; next_obs: STATE
    rewards, terminals = rewards[:, None], terminals[:, None]  # expand for correct broadcasting below

    stats.update(rm_stats)

    (policy_outputs, _), v_preds = self.model(obs)
    # assert isinstance(policy_outputs.base_dist, (TanhNormal, agents.sac.models.ConcatDirac))
    new_actions = policy_outputs.rsample()
    log_pi = policy_outputs.log_prob(new_actions)[:, None]
    # policy_mean = policy_outputs.base_dist.normal_mean
    # policy_log_std = torch.log(policy_outputs.base_dist.normal_std)
    assert log_pi.dim() == 2 and log_pi.shape[1] == 1, "use Independent(Normal(...), 1) instead of Normal(...)"

    # no alpha loss / adaptive entropy scaling

    # QF Loss
    # (pt, _), _ = self.model_target(obs)
    # new_actions_tgt = pt.sample()
    new_actions_tgt = struct(a=new_actions.a.detach()) if self.target_action_detach else struct(a=new_actions.a)

    new_next_obs = struct(next_obs, s=torch.cat((next_obs.s[:, :-actions.a.shape[1]], new_actions.a), 1))
    # new_next_obs_no_grad = struct(s=new_next_obs.s.detach())
    new_next_obs_tgt = struct(next_obs, s=torch.cat((next_obs.s[:, :-actions.a.shape[1]], new_actions_tgt.a), 1))
    _, v2s_target = self.model_target(new_next_obs_tgt)
    _, v2s = self.model_nograd(new_next_obs)

    # v2_target, _ = v2_target.min(1, keepdim=True)
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
    v_loss = sum(fu.mse_loss(v_pred, v_target) for v_pred in v_preds)
    # v_loss = fu.mse_loss(v_pred, v_target)

    # Policy Loss (with reparameterization)
    # policy_loss = self.entropy_scale * log_pi.mean() - self.outnorm.unnormalize(v2.mean())
    policy_loss = self.entropy_scale * log_pi.mean() - ((1. - terminals) * self.discount * self.outnorm.unnormalize(v2)).mean()
    policy_loss = self.outnorm.normalize(policy_loss)

    # Update Networks
    self.opt.zero_grad()
    # alpha = self.v_loss / (self.v_loss + 1)
    total_loss = self.loss_alpha * policy_loss + (1-self.loss_alpha) * v_loss
    total_loss.backward()
    self.opt.step()
    stats.update(total_loss=total_loss.detach())
    stats.update(loss_value=v_loss.detach())
    stats.update(loss_actor=policy_loss.detach())

    if self.num_updates % self.target_freq == 0:
      polyak(self.model_target, self.model, self.polyak)
      self.outnorm_target.m1 = self.polyak * self.outnorm_target.m1 + (1-self.polyak) * self.outnorm.m1
      self.outnorm_target.std = self.polyak * self.outnorm_target.std + (1 - self.polyak) * self.outnorm.std

    self.num_updates += 1

    stats.update(memory_size=len(self.memory), updates=self.num_updates, entropy_scale=self.entropy_scale)
    return ensure_detached(stats)
