import torch
import torch.nn as nn
from .util import mlp, resnet
from models.transport_small import TransportSmall
from utils import utils

class TwinQ(nn.Module):
    def __init__(self, state_dim, crop_size=64, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        
        self.q1 = TransportSmall(
            in_channels=3, 
            n_rotations=1, #8 
            crop_size=crop_size, 
            preprocess=utils.preprocess,
            verbose=False,
            name="Transport-Q1")
        self.q2 = TransportSmall(
            in_channels=3, 
            n_rotations=1, #8 
            crop_size=crop_size, 
            preprocess=utils.preprocess,
            verbose=False,
            name="Transport-Q2")

    def both(self, state, patch):
        return self.q1(state, patch), self.q2(state, patch)

    def forward(self, state, patch):
        return torch.min(*self.both(state, patch))
    

class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=16, n_hidden=2):
        super().__init__()
        self.v = resnet(
            num_blocks=4,
            in_channels=3,
            out_channels=1, # number of blocks
            hidden_channels=hidden_dim,
            output_activation=None,
            stride=1
            )

    def forward(self, state):
        return self.v(state)
    

# old code
# class TwinQ(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
#         super().__init__()
#         dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
#         self.q1 = mlp(dims, squeeze_output=True)
#         self.q2 = mlp(dims, squeeze_output=True)

#     def both(self, state, action):
#         sa = torch.cat([state, action], 1)
#         return self.q1(sa), self.q2(sa)

#     def forward(self, state, action):
#         return torch.min(*self.both(state, action))


# class ValueFunction(nn.Module):
#     def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
#         super().__init__()
#         dims = [state_dim, *([hidden_dim] * n_hidden), 1]
#         self.v = mlp(dims, squeeze_output=True)

#     def forward(self, state):
#         return self.v(state)