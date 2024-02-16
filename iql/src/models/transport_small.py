# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Transport module."""


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

#from src.models.resnet import ResNet43_8s, ResNet_small
from src.utils import utils, MeanMetrics, to_device
from src.utils.text import bold
from src.utils.utils import apply_rotations_to_tensor
from src.util import resnet #_strides


class TransportSmall(nn.Module):
    """Transport module."""
    def __init__(self, in_channels, n_rotations, crop_size, preprocess, verbose=False, name="Transport"):
        super().__init__()
        """Transport module for placing.

        Args:
          in_shape: shape of input image.
          n_rotations: number of rotations of convolving kernel.
          crop_size: crop size around pick argmax used as convolving kernel.
          preprocess: function to preprocess input images.
        """
        self.iters = 0
        self.n_rotations = n_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.preprocess = preprocess

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        # Crop before network (default for Transporters in CoRL submission).
        if not hasattr(self, 'output_dim'):
            self.output_dim = 3
        if not hasattr(self, 'kernel_dim'):
            self.kernel_dim = 3

        # 2 fully convolutional ResNets with 57 layers and 16-stride
        # self.model_query = ResNet_small(in_channels, self.output_dim)
        # self.model_key = ResNet_small(in_channels, self.kernel_dim)

        hidden_dim = 16
        self.model_query = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2*hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*hidden_dim),
            nn.ReLU(),
            nn.Conv2d(2*hidden_dim, 4*hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Conv2d(4*hidden_dim, 4*hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*hidden_dim),
            nn.ReLU(),
            nn.Conv2d(4*hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * hidden_dim),
            nn.ReLU(),
            nn.Conv2d(4 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
        )
        self.model_key = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * hidden_dim),
            nn.ReLU(),
            nn.Conv2d(2 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Conv2d(4 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * hidden_dim),
            nn.ReLU(),
            nn.Conv2d(4*hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * hidden_dim),
            nn.ReLU(),
            nn.Conv2d(4 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
        )
        # self.model_query = resnet(num_blocks=4, in_channels=3, out_channels=32, hidden_dim=16,
        #                                   output_activation=None)#, strides=[2, 2, 3, 3])
        # self.model_key = resnet(num_blocks=4, in_channels=3, out_channels=32, hidden_dim=16,
        #                                 output_activation=None)#, strides=[2, 2, 3, 3])

        self.device = to_device(
            [self.model_query, self.model_key], name, verbose=verbose)

        self.optimizer_query = optim.Adam(
            self.model_query.parameters(), lr=1e-4)
        self.optimizer_key = optim.Adam(self.model_key.parameters(), lr=1e-4)
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

        self.metric = MeanMetrics()

        self.softmax = nn.Softmax(dim=1)

        # if not self.six_dof:
        #   in0, out0 = ResNet43_8s(in_shape, output_dim, prefix="s0_")
        #   if self.crop_bef_q:
        #     # Passing in kernels: (64,64,6) --> (64,64,3)
        #     in1, out1 = ResNet43_8s(kernel_shape, kernel_dim, prefix="s1_")
        #   else:
        #     # Passing in original images: (384,224,6) --> (394,224,3)
        #     in1, out1 = ResNet43_8s(in_shape, output_dim, prefix="s1_")
        # else:
        #   in0, out0 = ResNet43_8s(in_shape, output_dim, prefix="s0_")
        #   # early cutoff just so it all fits on GPU.
        #   in1, out1 = ResNet43_8s(
        #       kernel_shape, kernel_dim, prefix="s1_", cutoff_early=True)

    # def set_bounds_pixel_size(self, bounds, pixel_size):
    #   self.bounds = bounds
    #   self.pixel_size = pixel_size

    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        # in0 = Rearrange('b h w c -> b c h w')(in0)
        # in1 = Rearrange('b h w c -> b c h w')(in1)

        output = F.conv2d(in0, in1, padding='same')
        # outputs = []
        # for b in range(in0.shape[0]):
        #     in0_b = in0[b:b+1, ...]
        #     in1_b = in1[b:b+1, ...]
        #     output_b = F.conv2d(in0_b, in1_b, padding='same')
        #     outputs.append(output_b)
        # output = torch.cat(outputs, dim=0)

        if softmax:
            output_shape = output.shape
            output = Rearrange('b c h w -> b (c h w)')(output)
            #output = output.clamp(min=1e-4)
            output = self.softmax(output)
            output = Rearrange(
                'b (c h w) -> b h w c',
                c=output_shape[1],
                h=output_shape[2],
                w=output_shape[3])(output)
            #output = output[0, ...]
            #output = output.detach().cpu().numpy()
        else:
            output = Rearrange('b c h w -> b h w c')(output)
        return output[:, :, :, 0]

    def forward(self, in_img, patch, softmax=False):
        # """Forward pass."""
        # img_unprocessed = np.pad(in_img, self.padding, mode='constant')
        # input_data = self.preprocess(img_unprocessed.copy())
        # input_data = Rearrange('h w c -> 1 h w c')(input_data)
        # in_tensor = torch.tensor(
        #     input_data, dtype=torch.float32
        # ).to(self.device)
        #
        # patch_unprocessed = np.pad(patch, self.padding, mode='constant')
        # patch_data = self.preprocess(patch_unprocessed.copy())
        # patch_data = Rearrange('h w c -> 1 h w c')(patch_data)
        # patch_tensor = torch.tensor(
        #     patch_data, dtype=torch.float32
        # ).to(self.device)
        #
        # # Rotate crop.
        # pivot = np.array([1, 1]) * self.crop_size//2 + self.pad_size
        # #pivot = list(np.array([p[1], p[0]]) + self.pad_size)
        #
        # # Crop before network (default for Transporters in CoRL submission).
        # crop = apply_rotations_to_tensor(
        #         patch_tensor, self.n_rotations, center=pivot
        #     )
        # crop = crop[:, pivot[0]-self.crop_size//2:pivot[0]+self.crop_size//2,
        #             pivot[1]-self.crop_size//2:pivot[1]+self.crop_size//2, :]
        # # crop = apply_rotations_to_tensor(
        # #     in_tensor, self.n_rotations, center=pivot)
        # # crop = crop[:, p[0]:(p[0] + self.crop_size),
        # #             p[1]:(p[1] + self.crop_size), :]
        in_tensor = self.preprocess(in_img).to(torch.float32).to(self.device)
        crop = self.preprocess(patch).to(torch.float32).to(self.device)
        in_tensor = Rearrange('b h w c -> b c h w')(in_tensor)
        crop = Rearrange('b h w c -> b c h w')(crop)
        # in_tensor = torch.tensor(in_img, dtype=torch.float32).to(self.device)
        # crop = torch.tensor(patch, dtype=torch.float32).to(self.device)
        logits = self.model_query(in_tensor)
        kernel = self.model_key(crop)
        logits = F.interpolate(logits, scale_factor=1/4, mode='bilinear')
        kernel = F.interpolate(kernel, scale_factor=1/4, mode='bilinear')

        # Crop after network (for receptive field, and more elegant).
        # logits, crop = self.model([in_tensor, in_tensor])
        # # crop = tf.identity(kernel_bef_crop)
        # crop = tf.repeat(crop, repeats=self.n_rotations, axis=0)
        # crop = tfa_image.transform(crop, rvecs, interpolation='NEAREST')
        # kernel_raw = crop[:, p[0]:(p[0] + self.crop_size),
        #                   p[1]:(p[1] + self.crop_size), :]

        # Obtain kernels for cross-convolution.
        # Padding of one on right and bottom of (h, w)
        # kernel_paddings = nn.ConstantPad2d((0, 0, 0, 1, 0, 1, 0, 0), 0)
        # kernel = kernel_paddings(kernel_raw)

        return self.correlate(logits, kernel, softmax)

    def train_block(self, in_img, patch, q, theta):
        output = self.forward(in_img, patch, softmax=False)

        itheta = theta / (2 * np.pi / self.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.n_rotations

        # Get one-hot pixel label map.
        label_size = in_img.shape[:2] + (self.n_rotations,)
        label = np.zeros(label_size)
        label[q[0], q[1], itheta] = 1

        # Get loss.
        label = Rearrange('h w c -> 1 (h w c)')(label)
        label = torch.tensor(label, dtype=torch.float32).to(self.device)
        label = torch.argmax(label, dim=1)
        output = Rearrange('b theta h w -> b (h w theta)')(output)

        loss = self.loss(output, label)

        return loss

    def train(self, in_img, patch, q, theta):
        """Transport patch to pixel q.

        Args:
          in_img: input image.
          patch: patch image
          q: pixel (y, x)
          theta: rotation label in radians.
          backprop: True if backpropagating gradients.

        Returns:
          loss: training loss.
        """

        self.metric.reset()
        self.train_mode()
        self.optimizer_query.zero_grad()
        self.optimizer_key.zero_grad()

        loss = self.train_block(in_img, patch, q, theta)
        loss.backward()
        self.optimizer_query.step()
        self.optimizer_key.step()
        self.metric(loss)

        self.iters += 1
        return np.float32(loss.detach().cpu().numpy())

    def test(self, in_img, patch, q, theta):
        """Test."""
        self.eval_mode()

        with torch.no_grad():
            loss = self.train_block(in_img, patch, q, theta)

        self.iters += 1
        return np.float32(loss.detach().cpu().numpy())

    def train_mode(self):
        self.model_query.train()
        self.model_key.train()

    def eval_mode(self):
        self.model_query.eval()
        self.model_key.eval()

    def format_fname(self, fname, is_query):
        suffix = 'query' if is_query else 'key'
        return fname.split('.pth')[0] + f'_{suffix}.pth'

    def load(self, fname, verbose):
        query_name = self.format_fname(fname, is_query=True)
        key_name = self.format_fname(fname, is_query=False)

        if verbose:
            device = "GPU" if self.device.type == "cuda" else "CPU"
            print(
                f"Loading {bold('transport query')} model on {bold(device)} from {bold(query_name)}")
            print(
                f"Loading {bold('transport key')}   model on {bold(device)} from {bold(key_name)}")

        self.model_query.load_state_dict(
            torch.load(query_name, map_location=self.device))
        self.model_key.load_state_dict(
            torch.load(key_name, map_location=self.device))

    def save(self, fname, verbose=False):
        query_name = self.format_fname(fname, is_query=True)
        key_name = self.format_fname(fname, is_query=False)

        if verbose:
            print(
                f"Saving {bold('transport query')} model to {bold(query_name)}")
            print(
                f"Saving {bold('transport key')}   model to {bold(key_name)}")

        torch.save(self.model_query.state_dict(), query_name)
        torch.save(self.model_key.state_dict(), key_name)
