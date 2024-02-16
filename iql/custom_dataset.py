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
        self.remove_bg = True
    
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
                        next_seg = os.path.join(trajectory_path, steps[i], 'seg_%s.npy'%self.view)
                        seg = os.path.join(trajectory_path, steps[i+1], 'seg_%s.npy'%self.view)
                        data_next_obj_info = os.path.join(trajectory_path, steps[i], 'obj_info.json')
                        obj_info = os.path.join(trajectory_path, steps[i + 1], 'obj_info.json')
                        data_rewards.append(reward)
                        data_terminals.append(terminal)
                        data_next_images.append(next_image)
                        data_images.append(image)
                        data_next_segs.append(next_seg)
                        data_segs.append(seg)
                        data_next_obj_infos.append(data_next_obj_info)
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

    def __getitem__(self, index):
        next_image = np.array(Image.open(self.data_next_images[index]))[:, :, :3]
        image = np.array(Image.open(self.data_images[index]))[:, :, :3]
        next_seg = np.load(self.data_next_segs[index])
        seg = np.load(self.data_segs[index])
        if self.remove_bg:
            next_image = next_image * (next_seg!=next_seg.max())[:, :, None]
            image = image * (seg!=seg.max())[:, :, None]
        reward = self.data_rewards[index]
        terminal = self.data_terminals[index]
        next_obj_info = json.load(open(self.data_next_obj_infos[index], 'r'))
        obj_info = json.load(open(self.data_obj_infos[index], 'r'))

        moved_object = self.find_object(next_obj_info, obj_info)
        action = self.find_action(moved_object, next_seg, seg)
        image_after_pick, patch = self.extract_patch(image, seg, moved_object)

        data = {
                'image': image/255.,
                'image_after_pick': image_after_pick/255.,
                'patch': patch/255.,
                'action': np.array(action),
                'next_image': next_image/255.,
                'reward': reward, 
                'terminal': terminal
                }
        return data

    def find_object(self, next_obj_info, obj_info):
        states = obj_info['state']
        next_states = next_obj_info['state']
        distances = []
        objects = list(states.keys())
        for o in objects:
            distance = np.linalg.norm(np.array(states[o][0]) - np.array(next_states[o][0]))
            distances.append(distance)
        moved_object = int(objects[np.argmax(distances)])
        return moved_object

    def find_action(self, moved_object, next_seg, seg):
        # seg_delta = np.abs(next_seg - seg)
        # unique, counts = np.unique(seg_delta[seg_delta != 0], return_counts=True)
        # moved_object = 8 - unique[counts.argmax()]
        next_patch_mask = (next_seg == moved_object).astype(float)
        pys, pxs = np.where(next_patch_mask == 1)
        py, px = np.mean(pys), np.mean(pxs)
        action = np.round([py, px]).astype(int).tolist()

        # convert action
        # 360 x 480 -> 10 x 13
        action[0] = int(action[0]/(360/10) - 0.5)
        action[1] = int(action[1]/(480/13) - 0.5)
        return action

    def extract_patch(self, image, seg, moved_object):
        image_after_pick = image * (seg != moved_object)[:, :, None]
        patch_mask = (seg == moved_object).astype(float)
        image_masked = image[:, :, :3] * patch_mask[:, :, None]

        cys, cxs = np.where(patch_mask)
        if not (len(cys)>0 and len(cxs)>0):
            print('empty patch')
            print()
        cy, cx = np.mean(cys), np.mean(cxs)
        yMin = int(cy - self.crop_size / 2)
        yMax = int(cy + self.crop_size / 2)
        xMin = int(cx - self.crop_size / 2)
        xMax = int(cx + self.crop_size / 2)

        image_size = image.shape
        patch = np.zeros([self.crop_size, self.crop_size, 3])
        patch[
        max(0, -yMin): max(0, -yMin) + min(image_size[0], yMax) - max(0, yMin),
        max(0, -xMin): max(0, -xMin) + min(image_size[1], xMax) - max(0, xMin),
        ] = image_masked[
            max(0, yMin): min(image_size[0], yMax),
            max(0, xMin): min(image_size[1], xMax),
            :3]
        patch = patch.astype(np.uint8)
        return image_after_pick, patch

    def __len__(self):
        return len(self.data_rewards)

