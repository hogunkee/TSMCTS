import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions.categorical import Categorical
from .models.transport_small import TransportSmall
from src.utils.utils import preprocess

from .util import mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

class DiscretePolicy(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        self.q = TransportSmall(
            in_channels=3, 
            n_rotations=1, #8 
            crop_size=crop_size, 
            preprocess=preprocess,
            verbose=False,
            name="Policy-Q")

    def forward(self, obs):
        state, patch = obs
        action_probs = self.q(state, patch, softmax=True)
        B, H, W = action_probs.size()
        C = 1
        action_probs_flatten = action_probs.view(B, -1)
        dist = Categorical(action_probs_flatten)
        actions_flatten = dist.sample()
        a0 = actions_flatten // (W * C)
        a1 = (actions_flatten % (W * C)) // C
        a2 = actions_flatten % C
        actions = torch.stack([a0, a1, a2], dim=-1)
        actions = actions.to(state.device)
        # actions = dist.sample().to(state.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        # z = action_probs == 0.0
        # z = z.float() * 1e-8
        action_probs = action_probs.clamp(min=1e-8, max=1.0)
        log_action_probs = torch.log(action_probs) # + z)

        return actions, action_probs, log_action_probs
    
    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs[0], obs[1])

class DeterministicPolicy(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        self.q = TransportSmall(
            in_channels=3, 
            n_rotations=1, #8 
            crop_size=crop_size, 
            preprocess=utils.preprocess,
            verbose=False,
            name="Policy-Q")

    def forward(self, obs):
        state, patch = obs
        action_probs = self.q(state, patch, softmax=True)
        B, H, W, C = action_probs.size()
        action_probs_flatten = action_probs.view(B, -1)
        actions_flatten = torch.argmax(action_probs_flatten, dim=-1)
        a0 = actions_flatten // (W * C)
        a1 = (actions_flatten % (W * C)) // C
        a2 = actions_flatten % C
        actions = torch.stack([a0, a1, a2], dim=-1)
        actions = actions.to(state.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return actions, action_probs, log_action_probs
    
    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs[0], obs[1])

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