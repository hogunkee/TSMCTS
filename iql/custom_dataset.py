import os
import numpy as np
import json
from PIL import Image

import torch
from torch.utils.data import Dataset

# B6/template_00001/traj_00092/001/
#       obj_info.json
#       rgb_front_top.png
#       rgb_top.png
#       depth_front_top.npy
#       depth_top.npy
#       seg_front_top.npy
#       seg_top.npy

class TabletopOfflineDataset(Dataset):
    def __init__(self, data_dir='/ssd/disk/TableTidyingUp/dataset_template/train', crop_size=160, view='top'):
        super().__init__()
        self.data_dir = data_dir
        self.crop_size = crop_size
        self.view = view
        self.get_data_paths()
    
    def get_data_paths(self):
        data_rewards = []
        data_terminals = []
        data_next_images = []
        data_images = []
        data_next_segs = []
        data_segs = []
        data_next_obj_infos = []
        data_obj_infos = []
        for scene in sorted(os.listdir(self.data_dir)):
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
                    rewards = [1.] + [0.] * (num_steps - 2)
                    terminals = [True] + [False] * (num_steps - 2)
                    for i in range(num_steps-1):
                        reward = rewards[i]
                        terminal = terminals[i]
                        next_image = os.path.join(trajectory_path, steps[i], 'rgb_%s.png'%self.view)
                        image = os.path.join(trajectory_path, steps[i+1], 'rgb_%s.png'%self.view)
                        next_seg = os.path.join(trajectory_path, steps[i], 'rgb_%s.png'%self.view)
                        seg = os.path.join(trajectory_path, steps[i+1], 'rgb_%s.png'%self.view)
                        next_obj_info = os.path.join(trajectory_path, steps[i], 'obj_info.json')
                        obj_info = os.path.join(trajectory_path, steps[i+1], 'obj_info.json')
                        data_rewards.append(reward)
                        data_terminals.append(terminal)
                        data_next_images.append(next_image)
                        data_images.append(image)
                        data_next_segs.append(next_seg)
                        data_segs.append(seg)
                        data_next_obj_infos.append(next_obj_info)
                        data_obj_infos.append(obj_info)
        self.data_rewards = data_rewards
        self.data_terminals = data_terminals
        self.data_next_images = data_next_images
        self.data_images = data_images
        self.data_next_segs = data_next_segs
        self.data_segs = data_segs
        self.data_next_obj_infos = data_next_obj_infos
        self.data_obj_infos = data_obj_infos
        return
    
    def __getitem__old__(self, index):
        data_path = self.data_paths[index]
        data_label = self.data_labels[index]

        if self.remove_bg:
            rgb = np.array(Image.open(os.path.join(data_path, 'rgb_%s.png'%self.view)))
            mask = np.load(os.path.join(data_path, 'seg_%s.npy'%self.view))
            rgb = rgb * (mask!=mask.max())[:, :, None]
        else:
            rgb = np.array(Image.open(os.path.join(data_path, 'rgb_%s.png'%self.view)))

        rgb = np.transpose(rgb[:, :, :3], [2, 0, 1])
        rgb = torch.from_numpy(rgb).type(torch.float)
        label = torch.from_numpy(np.array([data_label])).type(torch.float)
        return rgb, label

    def __getitem__(self, index):
        next_image = np.array(Image.open(self.data_next_images[index]))
        image = np.array(Image.open(self.data_images[index]))
        next_seg = np.load(self.data_next_segs[index])
        seg = np.load(self.data_segs[index])
        next_obj_info = json.load(open(self.data_next_obj_infos[index], 'r'))
        obj_info = json.load(open(self.data_obj_infos[index], 'r'))
        reward = self.data_rewards[index]
        terminal = self.data_terminals[index]

        target, patch_mask = self.find_object(next_seg, seg)
        action = self.find_action(next_obj_info, obj_info)
        image_after_pick, patch = self.extract_patch(image, patch_mask)

        data = {
                'image': image_after_pick,
                'patch': patch,
                'action': action, 
                'next_image': next_image, 
                'reward': reward, 
                'terminal': terminal
                }
        return data
    
    def find_object(self, next_seg, seg):
        return
    
    def find_action(self, next_obj_info, obj_info):
        return
    
    def extract_patch(self, image, patch_mask):
        return

    def __len__(self):
        return len(self.data_rewards)

