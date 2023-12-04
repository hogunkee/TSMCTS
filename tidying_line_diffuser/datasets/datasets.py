import numpy as np
from torch.utils.data import Dataset


class DiffusionDataset(Dataset):
    def __init__(self, data):
        self.data = np.reshape(data, (-1, 128, 128, 3))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class DiffusionDatasetNoBG(Dataset):
    def __init__(self, rgb, mask, resolution=128):
        self.rgb = np.reshape(rgb, (-1, resolution, resolution, 3))
        self.mask = np.reshape(mask, (-1, resolution, resolution))
        self.preprocess()

    def preprocess(self):
        mask = (self.mask!=0).astype(float)
        self.rgb = self.rgb * mask.reshape(-1, 128, 128, 1)
        self.mask = mask
    
    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, item):
        return self.rgb[item], self.mask[item]


class CondDiffusionDataset(Dataset):
    def __init__(self, rgb, segmap, resolution=128, cond_resolution=16):
        self.rgb = np.reshape(rgb, (-1, resolution, resolution, 3))
        self.segmap = np.reshape(segmap, (-1, cond_resolution, cond_resolution))

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, item):
        return self.rgb[item], self.segmap[item]


class CondDiffusionDatasetNoBG(Dataset):
    def __init__(self, rgb, segmap, mask, resolution=128, cond_resolution=16):
        self.rgb = np.reshape(rgb, (-1, resolution, resolution, 3))
        self.mask = np.reshape(mask, (-1, resolution, resolution))
        self.segmap = np.reshape(segmap, (-1, cond_resolution, cond_resolution))

    def preprocess(self):
        mask = (self.mask!=0).astype(float)
        self.rgb = self.rgb * mask
        self.mask = mask

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, item):
        return self.rgb[item], self.segmap[item]
