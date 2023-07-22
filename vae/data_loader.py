import os
import numpy as np
import torch
from torch.utils.data import Dataset


class UR5Dataset(Dataset):
    def __init__(self, data_dir='/home/gun/ssd/disk/ur5_tidying_data/3block'):
        super().__init__()
        self.data_dir = data_dir
        self.buff_i = None
        self.buff_d = None
        self.buff_p = None

        self.fsize = 2048
        self.find_urdata(self.data_dir)
        self.current_fidx = 0
        self.load_data(self.current_fidx)
    
    def __getitem__(self, index):
        npy_idx = (index // self.fsize) // self.num_file
        infile_idx = index % self.fsize
        if npy_idx != self.current_fidx:
            self.load_data(npy_idx)
            self.current_fidx = npy_idx
        i = self.buff_i[infile_idx]
        d = self.buff_d[infile_idx]
        p = self.buff_p[infile_idx]
        p = self.pos2pixel(p)
        i = np.transpose(i, [2, 0, 1])
        i = torch.from_numpy(i).type(torch.float)
        d = torch.from_numpy(d).type(torch.float)
        p = torch.from_numpy(p).type(torch.float)
        return i, d, p

    def __len__(self):
        return self.num_file * self.fsize

    def find_urdata(self, data_dir):
        rgb_list = [f for f in os.listdir(data_dir) if f.startswith('image_')]
        depth_list = [f for f in os.listdir(data_dir) if f.startswith('depth_')]
        pose_list = [f for f in os.listdir(data_dir) if f.startswith('pose_')]
        #rotation_list = [f for f in os.listdir(data_dir) if f.startswith('rotation_')]

        self.rgb_list = sorted(rgb_list)
        self.depth_list = sorted(depth_list)
        self.pose_list = sorted(pose_list)
        #self.rotation_list = sorted(rotation_list)

        self.num_file = len(self.rgb_list)

    def load_data(self, dnum):
        print('load %d-th npy file.' %dnum)
        self.buff_i = np.load(os.path.join(self.data_dir, self.rgb_list[dnum]))
        self.buff_d = np.load(os.path.join(self.data_dir, self.depth_list[dnum]))
        self.buff_p = np.load(os.path.join(self.data_dir, self.pose_list[dnum]))

    def pos2pixel(self, poses):
        x = poses[:, 0]
        y = poses[:, 1]

        theta = 30 * np.pi / 180
        cx, cy, cz = 0.0, 0.65, 1.75
        fovy = 45.0
        camera_height = camera_width = 96
        f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
        u0 = 0.5 * camera_width
        v0 = 0.5 * camera_height
        z0 = 0.9
        
        y_cam = np.cos(theta) * (y - cy - np.tan(theta) * (z0 - cz)) + 1e-10
        dv = f * np.cos(theta) / ((cz - z0) / y_cam - np.sin(theta))
        v = dv + v0
        u = - dv * x / y_cam + u0
        return np.concatenate([u[:, np.newaxis], v[:, np.newaxis]], axis=1)
        #return u, v
        #return int(np.round(u)), int(np.round(v))
