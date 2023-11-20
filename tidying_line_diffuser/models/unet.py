import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from einops import rearrange
import pdb
from torch.distributions import Bernoulli


from .helpers import (
    SinusoidalPosEmb,
    Downsample2d,
    Upsample2d,
    Conv2dBlock,
)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class ResidualBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5, mish=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv2dBlock(inp_channels, out_channels, kernel_size, mish),
            Conv2dBlock(out_channels, out_channels, kernel_size, mish),
        ])

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        if embed_dim > 0:
            self.time_mlp = nn.Sequential(
                act_fn,
                nn.Conv2d(embed_dim, out_channels, 1)
            )
        else:
            self.time_mlp = None

        self.residual_conv = nn.Conv2d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t=None):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        if t is not None:
            out = self.blocks[0](x) + self.time_mlp(t)
        else:
            out = self.blocks[0](x)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        kernel_size=5,
        condition_dropout=0.25
    ):
        super().__init__()

        dims = [input_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        mish = True
        act_fn = nn.Mish()

        self.time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        embed_dim = dim + cond_dim

        self.condition_dropout = condition_dropout
        self.dropout_mask = Bernoulli(1. - condition_dropout)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualBlock(dim_in, dim_out, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish),
                ResidualBlock(dim_out, dim_out, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish),
                Downsample2d(dim_out) if not is_last else nn.Identity(),
                Downsample2d(cond_dim) if not is_last else nn.Identity()
            ]))
        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualBlock(dim_out * 2, dim_in, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish),
                ResidualBlock(dim_in, dim_in, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish),
                Upsample2d(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Conv2dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv2d(dim, input_dim, 1),
        )

    def forward(self, x, cond, time, use_dropout=True, force_dropout=False):
        t = self.time_mlp(time)
        if use_dropout:
            mask = self.dropout_mask.sample((cond.shape[0], 1, 1, 1)).to(x.device)
            cond = cond * mask
        if force_dropout:
            cond = cond * 0

        h, h_cond = [], []

        for resnet, resnet2, downsample, downsample_cond in self.downs:
            B, _, W, H = cond.size()
            t_ = t.reshape(B, -1, 1, 1).repeat(1, 1, W, H)
            t_ = torch.cat([t_, cond], dim=1)
            x = resnet(x, t_)
            x = resnet2(x, t_)
            h.append(x)
            h_cond.append(t_)
            x = downsample(x)
            cond = downsample_cond(cond)

        B, _, W, H = cond.size()
        t_ = t.reshape(B, -1, 1, 1).repeat(1, 1, W, H)
        t_ = torch.cat([t_, cond], dim=1)
        x = self.mid_block1(x, t_)
        x = self.mid_block2(x, t_)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            t_ = h_cond.pop()
            x = resnet(x, t_)
            x = resnet2(x, t_)
            x = upsample(x)

        x = self.final_conv(x)

        return x
