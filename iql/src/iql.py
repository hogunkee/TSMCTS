import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from .util import DEFAULT_DEVICE, compute_batched, update_exponential_moving_average


EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class ImplicitQLearning(nn.Module):
    def __init__(self, qf, vf, policy, v_optimizer_factory, q_optimizer_factory, policy_optimizer_factory,
                 max_steps, tau, beta, discount=0.99, alpha=0.005):
        super().__init__()
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = v_optimizer_factory(self.vf.parameters())
        self.q_optimizer = q_optimizer_factory(self.qf.parameters())
        self.policy_optimizer = policy_optimizer_factory(self.policy.parameters())
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha

    def update(self, observations, actions, next_observations, rewards, terminals):
        B = actions.shape[0]
        with torch.no_grad():
            target_q_values = self.q_target(observations)
            target_q_a_values = target_q_values[torch.arange(B), actions[:, 0], actions[:, 1]]
            target_q_a_values = target_q_a_values.view(-1, 1)
            # target_q_a_values = []
            # for b, a in enumerate(actions):
            #     target_q_a_values.append(target_q_values[b, a[0], a[1]])
            # target_q_a_values = torch.stack(target_q_a_values).view(-1, 1)
            #target_q_a_values = target_q_values.gather(1, actions)
            next_v = self.vf(next_observations)

        # v, next_v = compute_batched(self.vf, [observations, next_observations])

        # Update value function
        v = self.vf(observations)
        adv = target_q_a_values - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        targets = rewards.view(-1, 1) + (1. - terminals.float().view(-1, 1)) * self.discount * next_v.detach()
        q1_values, q2_values = self.qf.both(observations)
        # q1_a_values = []
        # q2_a_values = []
        # for b, a in enumerate(actions):
        #     q1_a_values.append(q1_values[b, a[0], a[1]])
        #     q2_a_values.append(q2_values[b, a[0], a[1]])
        # q1_a_values = torch.stack(q1_a_values).view(-1, 1)
        # q2_a_values = torch.stack(q2_a_values).view(-1, 1)
        q1_a_values = q1_values[torch.arange(B), actions[:, 0], actions[:, 1]]
        q2_a_values = q2_values[torch.arange(B), actions[:, 0], actions[:, 1]]
        q1_a_values = q1_a_values.view(-1, 1)
        q2_a_values = q2_a_values.view(-1, 1)
        #qs = qs.gather(1, actions.unsqueeze(1).expand(-1, qs.size(1)))
        q_loss = (F.mse_loss(q1_a_values, targets) + F.mse_loss(q2_a_values, targets)) / 2
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # Update policy
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        _, _, log_action_probs = self.policy(observations)
        bc_losses = -log_action_probs[torch.arange(B), actions[:, 0], actions[:, 1]]
        #bc_losses = -log_action_probs.gather(1, actions)
        policy_loss = torch.mean(exp_adv * bc_losses)
        # if isinstance(policy_out, torch.distributions.Distribution):
        #     bc_losses = -policy_out.log_prob(actions)
        # elif torch.is_tensor(policy_out):
        #     assert policy_out.shape == actions.shape
        #     bc_losses = torch.sum((policy_out - actions)**2, dim=1)
        # else:
        #     raise NotImplementedError
        # policy_loss = torch.mean(exp_adv * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        return {'V-loss': v_loss.detach().cpu().numpy(), 'Q-loss': q_loss.detach().cpu().numpy(), 'Policy-loss': policy_loss.detach().cpu().numpy()}
