import os
import numpy as np
from PIL import Image
from tqdm import tqdm

data_dir = '/ssd/disk/ur5_tidying_data/pybullet_single_bg/images'

f_size = 900
save_dir = '/ssd/disk/ur5_tidying_data/pybullet_remove_bg/train'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

scene_list = sorted([f for f in os.listdir(data_dir) if f.startswith('rgb_')])
seg_list = sorted([f for f in os.listdir(data_dir) if f.startswith('seg_')])
print(len(scene_list), 'scenes exist.')

im_buffer = []
train_scene_list = scene_list[:4500]
train_seg_list = seg_list[:4500]
for i in tqdm(range(len(train_scene_list))):
    scene = train_scene_list[i]
    seg = train_seg_list[i]
    rgb_path = os.path.join(data_dir, scene)
    seg_path = os.path.join(data_dir, seg)
    im = Image.open(rgb_path)
    segmentation = np.flip(np.load(seg_path), 0)
    masked = np.array(im) * (segmentation!=0).reshape(500, 500, 1)
    masked[segmentation==0] = (255 * np.array([0.485, 0.456, 0.406, 1])).astype(np.uint8)
    im_resized = Image.fromarray(masked).resize((200, 200))
    im_norm = np.array(im_resized) / 255.0
    #im_norm = np.array(im_resized).transpose([2,0,1]) / 255.0
    im_buffer.append(im_norm)
    if len(im_buffer)==f_size:
        num_files = len([f for f in os.listdir(save_dir) if 'image_' in f])
        save_name = os.path.join(save_dir, 'image_%04d.npy' %num_files)
        np.save(save_name, np.array(im_buffer))
        im_buffer.clear()

f_size = 500
save_dir = '/ssd/disk/ur5_tidying_data/pybullet_remove_bg/test'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

im_buffer = []
test_scene_list = scene_list[4500:]
test_seg_list = seg_list[4500:]
for i in tqdm(range(len(test_scene_list))):
    scene = test_scene_list[i]
    seg = test_seg_list[i]
    rgb_path = os.path.join(data_dir, scene)
    seg_path = os.path.join(data_dir, seg)
    im = Image.open(rgb_path)
    segmentation = np.flip(np.load(seg_path), 0)
    masked = np.array(im) * (segmentation!=0).reshape(500, 500, 1)
    masked[segmentation==0] = (255 * np.array([0.485, 0.456, 0.406, 1])).astype(np.uint8)
    im_resized = Image.fromarray(masked).resize((200, 200))
    im_norm = np.array(im_resized) / 255.0
    #im_norm = np.array(im_resized).transpose([2,0,1]) / 255.0
    im_buffer.append(im_norm)
    if len(im_buffer)==f_size:
        num_files = len([f for f in os.listdir(save_dir) if 'image_' in f])
        save_name = os.path.join(save_dir, 'image_%04d.npy' %num_files)
        np.save(save_name, np.array(im_buffer))
        im_buffer.clear()

