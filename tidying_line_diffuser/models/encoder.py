import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F


from .unet import ResidualBlock
from .helpers import Downsample2d, Upsample2d, Conv2dBlock


class SquashedNormal(Normal):
    def __init__(self, loc, scale):
        super().__init__(loc, scale)
        self.squashed_mean = torch.tanh(loc)

    def sample(self, sample_shape=torch.Size()):
        x = super().sample(sample_shape)
        return torch.tanh(x)

    def rsample(self, sample_shape=torch.Size()):
        x = super().rsample(sample_shape)
        return torch.tanh(x)

    def log_prob(self, value):
        unsqueezed_x = torch.atanh(value.clamp(min=-0.999999, max=0.999999))
        jacob = 2 * (math.log(2) - unsqueezed_x - F.softplus(-2. * unsqueezed_x))
        log_prob = (super().log_prob(unsqueezed_x) - jacob)
        return log_prob

    def sample_with_logprob(self, sample_shape=torch.Size()):
        x = super().sample(sample_shape)
        jacob = 2 * (math.log(2) - x - F.softplus(-2. * x))
        log_prob = (super().log_prob(x) - jacob)
        return torch.tanh(x), log_prob

    def rsample_with_logprob(self, sample_shape=torch.Size()):
        x = super().rsample(sample_shape)
        jacob = 2 * (math.log(2) - x - F.softplus(-2. * x))
        log_prob = (super().log_prob(x) - jacob)
        return torch.tanh(x), log_prob

    def detach(self):
        return SquashedNormal(self.loc.detach(), self.scale.detach())

    @property
    def mean(self):
        return self.squashed_mean

    @property
    def unsquashed_mean(self):
        return self.loc


class Encoder(nn.Module):
    def __init__(
        self,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        output_dim=1,
        kernel_size=5,
    ):
        super().__init__()

        dims = [3, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        mish = True
        act_fn = nn.Mish()

        embed_dim = 0
        self.output_dim = output_dim

        self.downs = nn.ModuleList([])
        # self.ups = nn.ModuleList([])
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
        # self.mid_block2 = ResidualBlock(mid_dim, mid_dim, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish)
        self.final_conv = nn.Conv2d(mid_dim, 2 * output_dim, 1)

        # for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
        #     is_last = ind >= (num_resolutions - 1)
        #
        #     self.ups.append(nn.ModuleList([
        #         ResidualBlock(dim_out * 2, dim_in, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish),
        #         ResidualBlock(dim_in, dim_in, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish),
        #         Upsample2d(dim_in) if not is_last else nn.Identity()
        #     ]))

        # self.final_conv = nn.Sequential(
        #     Conv2dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
        #     nn.Conv2d(dim, output_dim, 1),
        # )

    def forward(self, x, compute_loss=True):
        # h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x)
            x = resnet2(x)
            # h.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        # x = self.mid_block2(x)

        # for resnet, resnet2, upsample in self.ups:
        #     x = torch.cat((x, h.pop()), dim=1)
        #     x = resnet(x)
        #     x = resnet2(x)
        #     x = upsample(x)

        x = self.final_conv(x)
        mean, log_std = torch.split(x, self.output_dim, dim=1)
        stddev = log_std.clamp(min=-20., max=2.).exp()
        posterior = SquashedNormal(mean, stddev)
        if compute_loss:
            prior = Normal(torch.zeros_like(mean), torch.ones_like(stddev))
            loss = kl_divergence(posterior, prior).mean()
        else:
            loss = None

        return posterior, loss


class Decoder(nn.Module):
    def __init__(
        self,
        dim=32,
        dim_mults=(8, 4, 2, 1),
        input_dim=1,
        kernel_size=5,
    ):
        super().__init__()

        dims = [input_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        mish = True
        act_fn = nn.Mish()

        embed_dim = 0

        # self.downs = nn.ModuleList([])
        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.blocks.append(nn.ModuleList([
                ResidualBlock(dim_in, dim_out, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish),
                ResidualBlock(dim_out, dim_out, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish),
                Upsample2d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish)
        # self.mid_block2 = ResidualBlock(mid_dim, mid_dim, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish)
        self.final_conv = nn.Conv2d(mid_dim, 3, 1)

        # for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
        #     is_last = ind >= (num_resolutions - 1)
        #
        #     self.ups.append(nn.ModuleList([
        #         ResidualBlock(dim_out * 2, dim_in, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish),
        #         ResidualBlock(dim_in, dim_in, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish),
        #         Upsample2d(dim_in) if not is_last else nn.Identity()
        #     ]))

        # self.final_conv = nn.Sequential(
        #     Conv2dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
        #     nn.Conv2d(dim, output_dim, 1),
        # )

    def forward(self, x):
        # h = []

        for resnet, resnet2, upsample in self.blocks:
            x = resnet(x)
            x = resnet2(x)
            # h.append(x)
            x = upsample(x)

        x = self.mid_block1(x)
        # x = self.mid_block2(x)

        # for resnet, resnet2, upsample in self.ups:
        #     x = torch.cat((x, h.pop()), dim=1)
        #     x = resnet(x)
        #     x = resnet2(x)
        #     x = upsample(x)

        x = self.final_conv(x)
        return torch.tanh(x)
