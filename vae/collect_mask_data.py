import argparse
import numpy as np
from backgroundsubtraction_module import BackgroundSubtraction

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='/home/gun/ssd/disk/ur5_tidying_data/3block/', type=str)
args = parser.parse_args()

data_dir = args.data_dir
n_objects = 3
save_freq = 2048

backsub = BackgroundSubtraction(pad=4)
backsub.fitting_model(os.path.join(data_dir, 'image_0.npy'))

buff_masks = []
buff_centers = []
f_list = [f for f in os.listdir(data_dir) if f.startswith('image_')]
for fname in f_list:
    images = np.load(f) * 255
    for img in images:
        masks, colors, fmask = backsub.get_masks(img, n_clusters=n_objects)
        centers = []
        for m in masks:
            sy, sx = np.where(m)
            y = np.round(np.mean(sy)).astype(int)
            x = np.round(np.mean(sx)).astype(int)
            centers.append([x, y])
        buff_masks.append(np.array(masks))
        buff_centers.append(np.array(centers))
        
        if len(buff_masks)%save_freq==0:
            m_filename = os.path.join(data_dir, fname.replace('image_', 'mask_'))
            c_filename = os.path.join(data_dir, fname.replace('image_', 'center_'))

            np.save(m_filename, np.array(buff_masks))
            np.save(c_filename, np.array(buff_centers))
            print('Saved %s.' %m_filename)

            buff_masks.clear()
            buff_centers.clear()

