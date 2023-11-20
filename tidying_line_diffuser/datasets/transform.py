import numpy as np
import torch
from torchvision.transforms.functional import affine
from torchvision.transforms import ColorJitter


class Transform:
    def __init__(self, max_translate=6, brightness=0.1, contrast=0.1, hue=0.02):
        self.color_jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            hue=hue
        )
        self.max_translate = max_translate

    def __call__(self, image):
        image_aug = self.color_jitter(image)
        image_aug = 2. * image_aug / 255. - 1.
        return image_aug
