import numpy as np
from torch.utils.data import Dataset


class DiffusionDataset(Dataset):
    def __init__(self, data):
        self.data = np.reshape(data, (-1, 128, 128, 3))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class CondDiffusionDataset(Dataset):
    def __init__(self, rgb, segmap, resolution=128, cond_resolution=16):
        self.rgb = np.reshape(rgb, (-1, resolution, resolution, 3))
        self.segmap = np.reshape(segmap, (-1, cond_resolution, cond_resolution))

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, item):
        return self.rgb[item], self.segmap[item]
