import numpy as np
from torch.utils.data import Dataset


class DiffusionDataset(Dataset):
    def __init__(self, data):
        self.data = np.reshape(data, (-1, 128, 128, 3))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
