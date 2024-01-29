import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PybulletNpyDataset(Dataset):
    def __init__(self, data_dir='/home/gun/ssd/disk/ur5_tidying_data/pybullet_line/train', augmentation=False, num_duplication=4):
        super().__init__()
        self.data_dir = data_dir
        self.augmentation = augmentation
        self.buff_i = None
        self.num_duplication = num_duplication
        self.fsize = 900

        self.find_npydata(self.data_dir)
        #self.current_fidx = 0
        self.load_data() #self.current_fidx)

        # soft labeling
        self.labels = np.linspace(1, 0, self.num_duplication)
    
    def __getitem__(self, index):
        npy_idx = (index // self.fsize) // self.num_file
        current_buff_i = self.buff_i[npy_idx]

        infile_idx = index % self.fsize
        i = current_buff_i[infile_idx]
        i = np.transpose(i, [2, 0, 1])
        i = torch.from_numpy(i).type(torch.float)

        # label
        label_idx = infile_idx % self.num_duplication
        label = torch.from_numpy(self.labels[label_idx:label_idx+1]).type(torch.float)
        return i, label

    def __len__(self):
        return self.num_file * self.fsize

    def find_npydata(self, data_dir):
        rgb_list = [f for f in os.listdir(data_dir) if f.startswith('image_')]
        self.rgb_list = sorted(rgb_list)
        self.num_file = len(self.rgb_list)

    def load_data(self):
        #print('load %d-th npy file.' %dnum)
        buff_i = []
        for rgb_file in self.rgb_list:
            patch_i = np.load(os.path.join(self.data_dir, rgb_file))[:, :, :, :3]
            buff_i.append(patch_i)
        self.buff_i = buff_i

