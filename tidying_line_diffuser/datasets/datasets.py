import numpy as np
from torch.utils.data import Dataset


class DiffusionDataset(Dataset):
    def __init__(self, data):
        self.data = np.reshape(data, (-1, 128, 128, 3))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], []


class DiffusionDatasetNoBG(Dataset):
    def __init__(self, rgb, mask, resolution=128):
        self.rgb = np.reshape(rgb, (-1, resolution, resolution, 3))
        self.mask = np.reshape(mask, (-1, resolution, resolution, 1))
        self.preprocess()

    def preprocess(self):
        mask = (self.mask!=0).astype(float)
        self.rgb = self.rgb * mask
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
        return self.rgb[item], self.segmap[item], []


class CondDiffusionDatasetNoBG(Dataset):
    def __init__(self, rgb, segmap, mask, resolution=128, cond_resolution=16):
        self.rgb = np.reshape(rgb, (-1, resolution, resolution, 3))
        self.mask = np.reshape(mask, (-1, resolution, resolution, 1))
        self.segmap = np.reshape(segmap, (-1, cond_resolution, cond_resolution))
        self.preprocess()

    def preprocess(self):
        mask = (self.mask!=0).astype(float)
        self.rgb = self.rgb * mask
        self.mask = mask

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, item):
        return self.rgb[item], self.segmap[item], self.mask[item]


class TargetCondDiffusionDataset(Dataset):
    def __init__(self, rgb, segmap, mask, resolution=128, cond_resolution=16, num_duplication=4):
        self.rgb = np.reshape(rgb, (-1, resolution, resolution, 3))
        self.mask = np.reshape(mask, (-1, resolution, resolution, 1))
        self.segmap = np.reshape(segmap, (-1, cond_resolution, cond_resolution))
        self.preprocess()

        self.num_duplication = num_duplication
        self.hash_augment = self.get_augmentation()
        self.fsize = len(self.hash_augment) * int(len(self.rgb) / num_duplication)

    def preprocess(self):
        mask = (self.mask!=0).astype(float)
        self.rgb = self.rgb * mask
        self.mask = mask

    def get_augmentation(self):
        num_duplication = self.num_duplication
        hash_list = []
        for i in range(num_duplication):
            for j in range(num_duplication):
                hash_list.append([i, j])
        return np.array(hash_list)

    def __len__(self):
        return self.fsize #len(self.rgb)

    def __getitem__(self, item):
        idx1 = item // len(self.hash_augment)
        idx2 = item % len(self.hash_augment)
        source_idx, target_idx = self.num_duplication * idx1 + self.hash_augment[idx2]
        return self.rgb[source_idx], self.rgb[target_idx], self.segmap[target_idx]
