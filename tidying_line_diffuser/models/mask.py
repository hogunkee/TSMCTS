import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F


from .unet import ResidualBlock
from .helpers import Downsample2d, Upsample2d, Conv2dBlock


class AttentionMask(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 4),
        input_dim=64,
        output_dim=16,
        kernel_size=5,
    ):
        super().__init__()

        dims = [input_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        mish = True
        act_fn = nn.Mish()

        embed_dim = 0

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResidualBlock(dim_in, dim_out, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish),
                ResidualBlock(dim_out, dim_out, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish),
                Downsample2d(dim_out) if not is_last else nn.Identity()
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
            nn.Conv2d(dim, output_dim, 1),
        )
        self.output_dim = output_dim

    def forward(self, x):
        B, C, W, H = x.size()
        h = []
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x)
            x = resnet2(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_block2(x)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x)
            x = resnet2(x)
            x = upsample(x)

        x = self.final_conv(x)
        x = F.softmax(x.reshape(-1, self.output_dim, W * H), dim=-1).reshape(-1, self.output_dim, W, H)

        return x