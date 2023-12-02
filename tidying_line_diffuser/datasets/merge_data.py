import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

res = 128
image_path = '/home/gun/ssd/disk/ur5_tidying_data/pybullet_single_bg/images/'
out_path = '../data/'
rgb_list = [f for f in os.listdir(image_path) if f.startswith('rgb_') and f.endswith('.png')]
seg_list = [f for f in os.listdir(image_path) if f.startswith('seg_') and f.endswith('.npy')]

rgb_npy_list = []
for _r in tqdm(rgb_list):
    rgb = np.array(Image.open(os.path.join(image_path, _r)))[:, :, :3]
    rgb_resized = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_AREA)
    rgb_npy_list.append(np.expand_dims(rgb_resized, 0))
rgb_concat = np.concatenate(rgb_npy_list)
print('save rgb data:', rgb_concat.shape)
np.save(os.path.join(out_path, 'rgb.npy'), rgb_concat)

seg_npy_list = []
for _s in tqdm(seg_list):
    segmap = np.load(os.path.join(image_path, _s))
    segmap = np.flip(segmap, 0)
    segmap_resized = cv2.resize(segmap, (res, res), interpolation=cv2.INTER_NEAREST)
    seg_npy_list.append(segmap_resized)
seg_concat = np.concatenate(np.expand_dims(seg_npy_list, 0))
print('save segmentation data:', seg_concat.shape)
np.save(os.path.join(out_path, 'segmap.npy'), seg_concat)
