import os
import numpy as np
import torch
from torch.utils.data import Dataset

def load_urdata(data_dir):
    rgb_list = [f for f in os.listdir(data_dir) if f.startswith('image_')]
    depth_list = [f for f in os.listdir(data_dir) if f.startswith('depth_')]
    pose_list = [f for f in os.listdir(data_dir) if f.startswith('pose_')]
    #rotation_list = [f for f in os.listdir(data_dir) if f.startswith('rotation_')]

    def load_numpy(fname_list):
        fpath_list = [os,path.join(datapath, f) for f in fname_list]
        data_list = []
        for f in fpath_list:
            data = np.load(f)
            data_list.append(data)
        return np.concatenate(data_list)

    images = load_numpy(rgb_list)
    depths = load_numpy(depth_list)
    poses = load_numpy(pose_list)
    return images, depths, poses

class UR5Dataset(Datset):
    def __init__(self, data_dir):
        super().__init__()
        images, depths, poses = load_urdata(data_dir)
        self.i = torch.tensor(i)
        self.d = torch.tensor(d)
        self.p = torch.tensor(p)
    
    def __getitem__(self, index):
        i = self.i[index]
        d = self.d[index]
        p = self.p[index]
        return i, d, p

    def __len__(self):
        return len(self.i)

