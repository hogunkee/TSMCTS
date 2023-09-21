import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TabletopDataset(Dataset):
    def __init__(self, data_dir='/home/gun/ssd/disk/tabletop_dataset_v5_public/training_set/', augmentation=False, num_duplication=5):
        super().__init__()
        self.data_dir = data_dir
        self.augmentation = augmentation
        self.num_duplication = num_duplication
        self.find_tabletopdata(self.data_dir)
    
    def __getitem__(self, index):
        file_idx = index % self.num_file
        # BGR image
        filename = self.rgb_list[file_idx]
        im = cv2.imread(filename)
        im_tensor = torch.from_numpy(im) / 255.0
        image_blob = im_tensor.permute(2, 0, 1).type(torch.float)
        return image_blob

    def __len__(self):
        return self.num_file

    def find_tabletopdata(self, data_dir):
        scene_list = sorted([s for s in os.listdir(self.data_dir) if s.startswith('scene_')])
        rgb_list = []
        depth_list = []
        seg_list = []
        for scene in scene_list:
            for i in range(1, self.num_duplication+1):
                rgb_list.append(os.path.join(self.data_dir, scene, 'rgb_%05d.jpeg'%i))
                depth_list.append(os.path.join(self.data_dir, scene, 'depth_%05d.png'%i))
                seg_list.append(os.path.join(self.data_dir, scene, 'segmentation_%05d.png'%i))

        self.rgb_list = rgb_list
        self.depth_list = depth_list
        self.seg_list = seg_list
        self.num_file = len(self.rgb_list)


class TabletopNpyDataset(Dataset):
    def __init__(self, data_dir='/home/gun/ssd/disk/ur5_tidying_data/tabletop_48x64', augmentation=False, num_duplication=5):
        super().__init__()
        self.data_dir = data_dir
        self.augmentation = augmentation
        self.buff_i = None
        self.num_duplication = num_duplication
        self.fsize = 2000

        self.find_tabletopdata(self.data_dir)
        self.current_fidx = 0
        self.load_data(self.current_fidx)
    
    def __getitem__(self, index):
        npy_idx = (index // self.fsize) // self.num_file
        if npy_idx != self.current_fidx:
            self.load_data(npy_idx)
            self.current_fidx = npy_idx

        infile_idx = index % self.fsize
        i = self.buff_i[infile_idx]
        i = np.transpose(i, [2, 0, 1])
        i = torch.from_numpy(i).type(torch.float)
        return i

    def __len__(self):
        return self.num_file * self.fsize

    def find_tabletopdata(self, data_dir):
        rgb_list = [f for f in os.listdir(data_dir) if f.startswith('image_')]
        self.rgb_list = sorted(rgb_list)
        self.num_file = len(self.rgb_list)

    def load_data(self, dnum):
        #print('load %d-th npy file.' %dnum)
        self.buff_i = np.load(os.path.join(self.data_dir, self.rgb_list[dnum]))


class UR5NpyDataset(Dataset):
    def __init__(self, data_dir='/home/gun/ssd/disk/ur5_tidying_data/3block', augmentation=False, num_duplication=4):
        super().__init__()
        self.data_dir = data_dir
        self.augmentation = augmentation
        self.buff_i = None
        self.num_duplication = num_duplication
        self.fsize = 2048
        if self.augmentation:
            self.buff_i_prime = None
            self.hash_augment = self.get_augmentation()
            self.fsize = len(self.hash_augment) * int(self.fsize / num_duplication)

        self.find_urdata(self.data_dir)
        self.current_fidx = 0
        self.load_data(self.current_fidx)
    
    def __getitem__(self, index):
        npy_idx = (index // self.fsize) // self.num_file
        if npy_idx != self.current_fidx:
            self.load_data(npy_idx)
            self.current_fidx = npy_idx

        if self.augmentation:
            file_idx = index % self.fsize
            idx1 = file_idx // len(self.hash_augment)
            idx2 = file_idx % len(self.hash_augment)
            infile_idx, infile_idx_prime = self.num_duplication * idx1 + self.hash_augment[idx2]

            i = self.buff_i[infile_idx]
            i_prime = self.buff_i[infile_idx_prime]
            i = np.transpose(i, [2, 0, 1])
            i_prime = np.transpose(i_prime, [2, 0, 1])
            i = torch.from_numpy(i).type(torch.float)
            i_prime = torch.from_numpy(i_prime).type(torch.float)
            return i, i_prime
        else:
            infile_idx = index % self.fsize

            i = self.buff_i[infile_idx]
            i = np.transpose(i, [2, 0, 1])
            i = torch.from_numpy(i).type(torch.float)
            return i

    def __len__(self):
        return self.num_file * self.fsize

    def find_urdata(self, data_dir):
        rgb_list = [f for f in os.listdir(data_dir) if f.startswith('image_')]
        self.rgb_list = sorted(rgb_list)
        self.num_file = len(self.rgb_list)

    def load_data(self, dnum):
        #print('load %d-th npy file.' %dnum)
        self.buff_i = np.load(os.path.join(self.data_dir, self.rgb_list[dnum]))

    def get_augmentation(self, num_duplication=4):
        hash_list = []
        for i in range(num_duplication):
            for j in range(i, num_duplication):
                hash_list.append([i, j])
        return np.array(hash_list)


class UR5Dataset(Dataset):
    def __init__(self, data_dir='/home/gun/ssd/disk/ur5_tidying_data/3block', augmentation=False, num_duplication=4):
        super().__init__()
        self.data_dir = data_dir
        self.augmentation = augmentation
        self.buff_i = None
        self.buff_d = None
        self.buff_p = None
        self.num_duplication = num_duplication
        self.fsize = 2048
        if self.augmentation:
            self.buff_i_prime = None
            self.buff_p_prime = None
            self.hash_augment = self.get_augmentation()
            self.fsize = len(self.hash_augment) * int(self.fsize / num_duplication)

        self.find_urdata(self.data_dir)
        self.current_fidx = 0
        self.load_data(self.current_fidx)
    
    def __getitem__(self, index):
        npy_idx = (index // self.fsize) // self.num_file
        if npy_idx != self.current_fidx:
            self.load_data(npy_idx)
            self.current_fidx = npy_idx

        if self.augmentation:
            file_idx = index % self.fsize
            idx1 = file_idx // len(self.hash_augment)
            idx2 = file_idx % len(self.hash_augment)
            infile_idx, infile_idx_prime = self.num_duplication * idx1 + self.hash_augment[idx2]

            i = self.buff_i[infile_idx]
            #d = self.buff_d[infile_idx]
            p = self.buff_p[infile_idx]
            i_prime = self.buff_i[infile_idx_prime]
            p_prime = self.buff_p[infile_idx_prime]
            p = self.pos2pixel(p)
            p_prime = self.pos2pixel(p_prime)
            i = np.transpose(i, [2, 0, 1])
            i_prime = np.transpose(i_prime, [2, 0, 1])
            i = torch.from_numpy(i).type(torch.float)
            #d = torch.from_numpy(d).type(torch.float)
            p = torch.from_numpy(p).type(torch.float)
            i_prime = torch.from_numpy(i_prime).type(torch.float)
            p_prime = torch.from_numpy(p_prime).type(torch.float)
            return i, p, i_prime, p_prime
        else:
            infile_idx = index % self.fsize

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
        #print('load %d-th npy file.' %dnum)
        self.buff_i = np.load(os.path.join(self.data_dir, self.rgb_list[dnum]))
        self.buff_d = np.load(os.path.join(self.data_dir, self.depth_list[dnum]))
        self.buff_p = np.load(os.path.join(self.data_dir, self.pose_list[dnum]))

    def get_augmentation(self, num_duplication=4):
        hash_list = []
        for i in range(num_duplication):
            for j in range(i, num_duplication):
                hash_list.append([i, j])
        return np.array(hash_list)

    def pos2pixel(self, poses):
        x = poses[:, 0]
        y = poses[:, 1]

        # rlview 2 #
        theta = 0
        cx, cy, cz = 0.0, 0.1, 1.8
        # rivlew #
        #theta = 30 * np.pi / 180
        #cx, cy, cz = 0.0, 0.65, 1.75
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

class PybulletNpyDataset(Dataset):
    def __init__(self, data_dir='/home/gun/ssd/disk/ur5_tidying_data/pybullet_line/train', augmentation=False, num_duplication=4):
        super().__init__()
        self.data_dir = data_dir
        self.augmentation = augmentation
        self.buff_i = None
        self.num_duplication = num_duplication
        self.fsize = 2000

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
        label = self.labels[label_idx]
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
            patch_i = np.load(os.path.join(self.data_dir, rgb_file))
            buff_i.append(patch_i)
        self.buff_i = buff_i

