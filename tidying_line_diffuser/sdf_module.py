import os
import sys
import cv2
import copy
import torch
import numpy as np

file_path = os.path.dirname(os.path.abspath(__file__))
ucn_path = os.path.join(file_path, '../../ur5_manipulation', 'UnseenObjectClustering')
sys.path.append(ucn_path)
import networks
from fcn.test_dataset import clustering_features
from fcn.config import cfg_from_file

from matplotlib import pyplot as plt

#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UCNModule():
    def __init__(self, using_depth=False):
        self.using_depth = using_depth
        if self.using_depth:
            self.pretrained = os.path.join(ucn_path, 'experiments/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth')
            self.cfg_file = os.path.join(ucn_path, 'experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml')
        else:
            self.pretrained = os.path.join(ucn_path, 'experiments/checkpoints/seg_resnet34_8s_embedding_cosine_color_sampling_epoch_16.checkpoint.pth')
            self.cfg_file = os.path.join(ucn_path, 'experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml')
        cfg_from_file(self.cfg_file)

        self.network_name = 'seg_resnet34_8s_embedding'
        self.network_data = torch.load(self.pretrained)
        self.network = networks.__dict__[self.network_name](2, 64, self.network_data).to(device)
        self.network.eval()

        self.network_crop = None
        #self.depth_bg = np.load(os.path.join(file_path, '../', 'ur5_mujoco/depth_bg_480.npy'))
        #self.params = self.get_camera_params()

    def eval_ucn(self, rgb, depth, data_format='HWC'):
        if data_format=='HWC':
            rgb = rgb.transpose([2, 0, 1])
        rgb_tensor = torch.Tensor(rgb).unsqueeze(0).to(device)

        if self.using_depth:
            xyz_img = self.process_depth(depth)
            depth_tensor = torch.Tensor(xyz_img).permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            depth_tensor = None

        features = self.network(rgb_tensor, None, depth_tensor).detach()
        out_label, selected_pixels = clustering_features(features[:1], num_seeds=100)

        features = features.cpu().numpy()[0]
        segmap = out_label.cpu().detach().numpy()[0]
        num_blocks = int(segmap.max())

        # get masks #
        masks = []
        for nb in range(1, num_blocks+1):
            _mask = (segmap == nb).astype(float)
            masks.append(_mask)

        # get bounding boxes #
        bboxes = []
        for m in masks:
            sy, sx = np.array(np.where(m)).min(1)
            my, mx = np.array(np.where(m)).max(1)
            dy = my - sy
            dx = mx - sx
            bbox = (sx, sy, dx, dy)
            bboxes.append(bbox)
        return bboxes, masks, segmap, features

    def get_camera_params(self):
        params = {}
        params['img_width'] = 480
        params['img_height'] = 480

        fovy = 45
        f = 0.5 * params['img_height'] / np.tan(fovy * np.pi / 360)
        params['fx'] = f
        params['fy'] = f
        return params

    def compute_xyz(self, depth):
        if 'fx' in self.params and 'fy' in self.params:
            fx = self.params['fx']
            fy = self.params['fy']
        else:
            aspect_ratio = self.params['img_width'] / self.params['img_height']
            e = 1 / (np.tan(np.radians(self.params['fov']/2.)))
            t = self.params['near'] / e
            b = -t
            r = t* aspect_ratio 
            l = -r
            alpha = self.params['img_width'] / (r - l)
            focal_length = self.params['near'] * alpha
            fx = focal_length
            fy = focal_length

        if 'x_offset' in self.params and 'y_offset' in self.params:
            x_offset = self.params['x_offset']
            y_offset = self.params['y_offset']
        else:
            x_offset = self.params['img_width'] / 2
            y_offset = self.params['img_height'] / 2

        indices = np.indices((self.params['img_height'], self.params['img_width']), dtype=np.float32).transpose(1,2,0)
        z_e = depth
        x_e = (indices[..., 1] - x_offset) * z_e / fx
        y_e = (indices[..., 0] - y_offset) * z_e / fy
        xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # shape: [H, W, 3]

        return xyz_img

    def process_depth(self, depth):
        data_augmentation = False
        if data_augmentation:
            pass
            #depth = augmentation.add_noise_to_depth(depth, self.params)
            #depth = augmentation.dropout_random_ellipses(depth, self.params)
        xyz_img = self.compute_xyz(depth)
        if data_augmentation:
            pass
            #xyz_img = augmantation.add_noise_to_xyz(xyz_img, depth, self.params)
        return xyz_img

if __name__=='__main__':
    from PIL import Image
    img_path = '/ssd/disk/ur5_tidying_data/pybullet_single_bg2/images/00006.png'
    x = np.array(Image.open(img_path))[:, :, :3] / 255.
    res = 250
    x = cv2.resize(x, (res, res), interpolation=cv2.INTER_AREA)

    ucn = UCNModule(using_depth=False)
    results = ucn.eval_ucn(x, None)
    print(results[1], 'objects detected.')
    plt.imshow(results[2])
    plt.show()
    print(results)
