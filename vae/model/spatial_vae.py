"""
This module implements, using PyTorch, the deep spatial autoencoder architecture presented in [1].
References:
    [1]: "Deep Spatial Autoencoders for Visuomotor Learning"
    Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, Pieter Abbeel
    Available at: https://arxiv.org/pdf/1509.06113.pdf
    [2]: https://github.com/tensorflow/tensorflow/issues/6271
"""

import torch
from torch import nn
from torch.nn import functional as F

class CoordinateUtils(object):
    @staticmethod
    def get_image_coordinates(pixel_source, res_source, res_target):
        pixel_norm = (pixel_source / (res_source - 1)) * 2 - 1
        pixel_target = (pixel_norm + 1) * (res_target - 1) / 2
        return pixel_target

    def normalise(pixel_source, res_source):
        pixel_norm = (pixel_source / (res_source - 1)) * 2 - 1
        return pixel_norm

    def denormalise(pixel_norm, res_target):
        pixel_target = (pixel_norm + 1) * (res_target - 1) / 2
        return pixel_target


class LookupUtils(object):
    @staticmethod
    def get_lookup(pixels, h, w, normalised=True):
        n = len(pixels)
        if normalised:
            pixels = CoordinateUtils.denormalise(pixels, h)
        lookup_filter = torch.zeros(n, h, w)
        for i in range(n):
            x, y = pixels[i].type(torch.int32)
            lookup_filter[y, x] = 1.0
        return lookup_filter
            

class CustomEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dims=[32, 64, 128, 256]):
        super().__init__()

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU()
                        )
                    )
            in_channels = h_dim
        self.cnn = nn.Sequential(*modules)

    def forward(self, x, p):
        out_cnn = self.cnn(x)
        _, c, h, w = out_cnn.shape
        lookup_filter = LookupUtils.get_lookup(p, h, w, normalised=True)
        out = F.conv2d(out_cnn, lookup_filter, padding=0)
        return out


class CustomDecoder(nn.Module):
    def __init__(self, latent_dimension, latent_height=6, latent_width=6, 
                 hidden_dims=[256, 128, 64, 32], out_channels=3, normalise=True):
        super().__init__()
        self.n_hidden = hidden_dims[0]
        self.latent_height = latent_height
        self.latent_width = latent_width

        self.linear = nn.Linear(in_features=latent_dimension, 
                                 out_features=latent_height * latent_width * hidden_dims[0])
        self.leaky_relu = nn.LeakyReLU()

        modules = []
        for i in range(len(hidden_dims)-1):
            modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i+1],
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i+1]),
                        nn.LeakyReLU()
                        )
                    )
        self.cnn_transpose = nn.Sequential(*modules)
        self.activate = nn.Tanh if normalise else nn.Sigmoid
        self.final_layer = nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[-1],
                                           hidden_dims[-1],
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1),
                        nn.BatchNorm2d(hidden_dims[-1]),
                        nn.LeakyReLU(),
                        nn.Conv2d(hidden_dims[-1], out_channels=out_channels,
                                  kernel_size=3, padding=1),
                        self.activate()
                        )

    def forward(self, z):
        out_linear = self.leaky_relu(self.linear(z))
        out_reshape = out_linear.view(-1, self.n_hidden, self.latent_height, self.latent_width)
        out_cnn_transpose = self.cnn_transpose(out_reshape)
        out = self.final_layer(out_cnn_transpose)
        return out


class CustomDeepSpatialAutoencoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[32, 64, 128, 32], latent_dimension=64, 
                 latent_height=6, latent_width=6, out_channels=3):
        """
        Same as DeepSpatialAutoencoder, but with your own custom modules
        :param encoder: Encoder
        :param decoder: Decoder
        """
        super().__init__()
        self.encoder = CustomEncoder(in_channels=in_channels, hidden_dims=hidden_dims)
        hidden_dims.reverse()
        self.decoder = CustomDecoder(latent_dimension=latent_dimension, latent_height=latent_height,
                             latent_width=latent_width, hidden_dims=hidden_dims, out_channels=out_channels)

    def forward(self, x):
        spatial_features = self.encoder(x)
        n, c, _2 = spatial_features.size()
        # (N, C * 2 = latent dimension)
        return self.decoder(spatial_features.view(n, c * 2))


class SVAE_Loss(object):
    def __init__(self):
        self.mse_loss = nn.MSELoss(reduction="sum")

    def __call__(self, reconstructed, target):
        loss = self.mse_loss(reconstructed, target)
        return loss
