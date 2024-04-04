import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions.categorical import Categorical
from .models.transport_small import TransportSmall
from .value_functions import ResNetQ, ResNetP

from .util import mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class PickPolicy(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        self.p = ResNetP(hidden_dim=32)

    def forward(self, obs):
        state_v, state_q, _ = obs
        N = state_q.size(0)
        state_v = state_v.unsqueeze(0).repeat([N, 1, 1, 1])
        state = torch.cat([state_v, state_q], dim=-1)

        action_probs = self.p(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        action_probs = action_probs.clamp(min=1e-8, max=1.0)
        log_action_probs = torch.log(action_probs) # + z)
        return action, action_probs, log_action_probs, dist.log_prob(action)

    def get_prob(self, obs):
        state_v, state_q, _ = obs
        N = state_q.size(0)
        state_v = state_v.unsqueeze(0).repeat([N, 1, 1, 1])
        state = torch.cat([state_v, state_q], dim=-1)

        action_probs = self.p(state)
        #action_probs = action_probs.clamp(min=1e-8, max=1.0)
        return action_probs
    
    
class DiscreteTransportPolicy(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        self.q = TransportSmall(
            in_channels=3, 
            n_rotations=1, #8 
            crop_size=crop_size, 
            verbose=False,
            name="Policy-Q")

    def forward(self, obs):
        _, state_q, patch = obs
        action_probs = self.q(state_q, patch, softmax=True)
        B, H, W = action_probs.size()
        C = 1
        action_probs_flatten = action_probs.view(B, -1)
        dist = Categorical(action_probs_flatten)
        actions_flatten = dist.sample()
        a0 = actions_flatten // (W * C)
        a1 = (actions_flatten % (W * C)) // C
        a2 = actions_flatten % C
        actions = torch.stack([a0, a1, a2], dim=-1)
        actions = actions.to(state_q.device)
        # actions = dist.sample().to(state.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        # z = action_probs == 0.0
        # z = z.float() * 1e-8
        action_probs = action_probs.clamp(min=1e-8, max=1.0)
        log_action_probs = torch.log(action_probs) # + z)

        return actions, action_probs, log_action_probs, dist.log_prob(actions_flatten)
    

class DeterministicTransportPolicy(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        self.q = TransportSmall(
            in_channels=3, 
            n_rotations=1, #8 
            crop_size=crop_size, 
            verbose=False,
            name="Policy-Q")

    def forward(self, obs):
        _, state_q, patch = obs
        action_probs = self.q(state_q, patch, softmax=True)
        B, H, W = action_probs.size()
        C = 1
        action_probs_flatten = action_probs.view(B, -1)
        actions_flatten = torch.argmax(action_probs_flatten, dim=-1)
        a0 = actions_flatten // (W * C)
        a1 = (actions_flatten % (W * C)) // C
        a2 = actions_flatten % C
        actions = torch.stack([a0, a1, a2], dim=-1)
        actions = actions.to(state_q.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return actions, action_probs, log_action_probs, dist.log_prob(actions_flatten)
    

class DiscreteResNetPolicy(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        self.q = ResNetQ(hidden_dim=32)

    def forward(self, obs):
        _, state_q, patch = obs
        q = self.q(state_q, patch)
        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)

        dist = Categorical(action_probs_flatten)
        actions_flatten = dist.sample()
        a0 = actions_flatten // W
        a1 = actions_flatten % W
        actions = torch.stack([a0, a1], dim=-1)
        actions = actions.to(state_q.device)
        # actions = dist.sample().to(state.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        # z = action_probs == 0.0
        # z = z.float() * 1e-8
        action_probs = action_probs.clamp(min=1e-8, max=1.0)
        log_action_probs = torch.log(action_probs) # + z)

        return actions, action_probs, log_action_probs, dist.log_prob(actions_flatten)

    def get_prob(self, obs):
        _, state_q, patch = obs
        q = self.q(state_q, patch)
        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)
        #action_probs = action_probs.clamp(min=1e-8, max=1.0)
        return action_probs
    
class DeterministicResNetPolicy(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        self.q = ResNetQ(hidden_dim=32)

    def forward(self, obs):
        _, state_q, patch = obs
        q = self.q(state_q, patch)
        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)
        
        actions_flatten = torch.argmax(action_probs_flatten, dim=-1)
        a0 = actions_flatten // W
        a1 = actions_flatten % W
        actions = torch.stack([a0, a1], dim=-1)
        actions = actions.to(state_q.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return actions, action_probs, log_action_probs, dist.log_prob(actions_flatten)
    
    def get_prob(self, obs):
        _, state_q, patch = obs
        q = self.q(state_q, patch)
        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)
        #action_probs = action_probs.clamp(min=1e-8, max=1.0)
        return action_probs
    
class GaussianPolicy(nn.Module):
    def __init__():
        super().__init__()

    def forward(self, obs):
        return

# class GaussianPolicy(nn.Module):
#     def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
#         super().__init__()
#         self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
#         self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

#     def forward(self, obs):
#         mean = self.net(obs)
#         std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
#         scale_tril = torch.diag(std)
#         return MultivariateNormal(mean, scale_tril=scale_tril)
#         # if mean.ndim > 1:
#         #     batch_size = len(obs)
#         #     return MultivariateNormal(mean, scale_tril=scale_tril.repeat(batch_size, 1, 1))
#         # else:
#         #     return MultivariateNormal(mean, scale_tril=scale_tril)

#     def act(self, obs, deterministic=False, enable_grad=False):
#         with torch.set_grad_enabled(enable_grad):
#             dist = self(obs)
#             return dist.mean if deterministic else dist.sample()


# class DeterministicPolicy(nn.Module):
#     def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
#         super().__init__()
#         self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
#                        output_activation=nn.Tanh)

#     def forward(self, obs):
#         return self.net(obs)

#     def act(self, obs, deterministic=False, enable_grad=False):
#         with torch.set_grad_enabled(enable_grad):
#             return self(obs)
