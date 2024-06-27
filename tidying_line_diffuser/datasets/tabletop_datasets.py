import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class TabletopDataset(Dataset):
    def __init__(self, data_dir='/ssd/disk/TableTidyingUp/dataset_template/train', remove_bg=True, label_type='linspace', view='top', get_mask=False):
        super().__init__()
        self.data_dir = data_dir
        self.remove_bg = remove_bg
        self.label_type = label_type
        self.get_mask = get_mask
        self.view = view
        self.data_paths, self.data_labels = self.get_data_paths()
    
    def get_data_paths(self):
        data_paths = []
        data_labels = []
        scene_list = sorted(os.listdir(self.data_dir))
        for scene in scene_list:
            scene_path = os.path.join(self.data_dir, scene)
            for template in sorted(os.listdir(scene_path)):
                template_path = os.path.join(scene_path, template)
                trajectories = sorted(os.listdir(template_path))
                for trajectory in trajectories:
                    trajectory_path = os.path.join(template_path, trajectory)
                    steps = sorted(os.listdir(trajectory_path))
                    num_steps = len(steps)
                    if num_steps!=5:
                        print('skip %s (%d steps)'%(trajectory_path, num_steps))
                        continue
                    if self.label_type == 'linspace':
                        labels = np.linspace(1, 0, num_steps)
                    elif self.label_type == 'binary':
                        labels = [1] + [0] * (num_steps - 1)
                    for i, step in enumerate(steps):
                        data_path = os.path.join(trajectory_path, step)
                        data_paths.append(data_path)
                        data_labels.append(labels[i])

        return data_paths, data_labels

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        data_label = self.data_labels[index]
        rgb = np.array(Image.open(os.path.join(data_path, 'rgb_%s.png'%self.view)))
        if self.remove_bg:
            mask = np.load(os.path.join(data_path, 'seg_%s.npy'%self.view))
            rgb = rgb * (mask!=mask.max())[:, :, None]

        #rgb = np.transpose(rgb[:, :, :3], [2, 0, 1]) / 255.
        rgb = rgb[:, :, :3] / 255.
        rgb = torch.from_numpy(rgb).type(torch.float)
        label = torch.from_numpy(np.array([data_label])).type(torch.float)
        if self.get_mask:
            mask = torch.from_numpy((mask!=mask.max())).type(torch.float)
            return mask, label, rgb
        else:
            return rgb, label

    def __len__(self):
        return len(self.data_paths)

class TabletopDiffusionDataset(TabletopDataset):
    def __init__(self, data_dir='/ssd/disk/TableTidyingUp/dataset_template/train', remove_bg=True):
        super().__init__(data_dir, remove_bg=remove_bg)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        data_label = self.data_labels[index]
        rgb = np.array(Image.open(os.path.join(data_path, 'rgb_%s.png'%self.view)))
        if self.remove_bg:
            mask = np.load(os.path.join(data_path, 'seg_%s.npy'%self.view))
            rgb = rgb * (mask!=mask.max())[:, :, None]

        #rgb = np.transpose(rgb[:, :, :3], [2, 0, 1]) / 255.
        rgb = rgb[:, :, :3] / 255.
        rgb = torch.from_numpy(rgb).type(torch.float)
        label = torch.from_numpy(np.array([data_label])).type(torch.float)
        if self.remove_bg:
            mask = torch.from_numpy((mask!=mask.max())).type(torch.float)
            return rgb, mask
        else:
            return rgb, []

class CondTabletopDiffusionDataset(TabletopDataset):
    def __init__(self, data_dir='/ssd/disk/TableTidyingUp/dataset_template/train', remove_bg=True):
        super().__init__(data_dir, remove_bg=remove_bg)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        data_label = self.data_labels[index]
        rgb = np.array(Image.open(os.path.join(data_path, 'rgb_%s.png'%self.view)))
        mask = np.load(os.path.join(data_path, 'seg_%s.npy'%self.view))
        if self.remove_bg:
            rgb = rgb * (mask!=mask.max())[:, :, None]

        #rgb = np.transpose(rgb[:, :, :3], [2, 0, 1]) / 255.
        rgb = rgb[:, :, :3] / 255.
        rgb = torch.from_numpy(rgb).type(torch.float)
        label = torch.from_numpy(np.array([data_label])).type(torch.float)

        segmap = mask * (mask!=mask.max())
        segmap = torch.from_numpy(segmap).type(torch.float)
        if self.remove_bg:
            mask = torch.from_numpy((mask!=mask.max())).type(torch.float)
            return rgb, segmap, mask
        else:
            return rgb, segmap, []

class TargetTabletopDiffusionDataset(TabletopDataset):
    def __init__(self, data_dir='/ssd/disk/TableTidyingUp/dataset_template/train', remove_bg=True, num_duplication=5):
        super().__init__(data_dir, remove_bg=remove_bg)
        self.num_duplication = num_duplication
        self.hash_augment = self.get_augmentation()
        self.fsize = len(self.hash_augment) * int(len(self.data_paths) / num_duplication)

    def get_augmentation(self):
        num_duplication = self.num_duplication
        hash_list = []
        for i in range(num_duplication):
            for j in range(num_duplication):
                hash_list.append([i, j])
        return np.array(hash_list)

    def __len__(self):
        return self.fsize #len(self.data_paths)

    def __getitem__(self, index):
        idx1 = index // len(self.hash_augment)
        idx2 = index % len(self.hash_augment)
        source_idx, target_idx = self.num_duplication * idx1 + self.hash_augment[idx2]

        src_data_path = self.data_paths[source_idx]
        tar_data_path = self.data_paths[target_idx]
        src_rgb = np.array(Image.open(os.path.join(src_data_path, 'rgb_%s.png'%self.view)))
        tar_rgb = np.array(Image.open(os.path.join(tar_data_path, 'rgb_%s.png'%self.view)))
        src_mask = np.load(os.path.join(src_data_path, 'seg_%s.npy'%self.view))
        tar_mask = np.load(os.path.join(tar_data_path, 'seg_%s.npy'%self.view))
        if self.remove_bg:
            src_rgb = src_rgb * (src_mask!=src_mask.max())[:, :, None]
            tar_rgb = tar_rgb * (tar_mask!=tar_mask.max())[:, :, None]

        #src_rgb = np.transpose(src_rgb[:, :, :3], [2, 0, 1]) / 255.
        #tar_rgb = np.transpose(tar_rgb[:, :, :3], [2, 0, 1]) / 255.
        src_rgb = src_rgb[:, :, :3] / 255.
        tar_rgb = tar_rgb[:, :, :3] / 255.
        src_rgb = torch.from_numpy(src_rgb).type(torch.float)
        tar_rgb = torch.from_numpy(tar_rgb).type(torch.float)

        src_segmap = src_mask * (src_mask!=src_mask.max())
        tar_segmap = tar_mask * (tar_mask!=tar_mask.max())
        src_segmap = torch.from_numpy(src_segmap).type(torch.float)
        tar_segmap = torch.from_numpy(tar_segmap).type(torch.float)
        #src_mask = torch.from_numpy((src_mask!=src_mask.max())).type(torch.float)
        #tar_mask = torch.from_numpy((tar_mask!=tar_mask.max())).type(torch.float)
        return src_rgb, src_segmap, tar_rgb, tar_segmap
