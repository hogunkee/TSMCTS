import os
import cv2
import numpy as np
from PIL import Image
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering

class BackgroundSubtraction():
    def __init__(self):
        self.pad = 10
        # self.model = None
        # self.fitting_model()
        self.model = cv2.createBackgroundSubtractorMOG2(detectShadows=False) #True

        #self.workspace_seg = None
        #self.make_empty_workspace_seg()

    def fitting_model_from_data(self, data_path, res=96):
        data_list = os.listdir(data_path)
        for data in data_list:
            if not data.endswith('.png'):
                continue
            print(data)
            im = np.array(Image.open(os.path.join(data_path, data)))/255.
            im = cv2.resize(im, (res, res), interpolation=cv2.INTER_AREA)
            im = (im*255).astype(np.uint8)
            self.model.apply(im)

    def fitting_model(self):
        self.model = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        pad = self.pad 
        frame = (np.load(os.path.join(FILE_PATH, '../dqn_image/scenes/bg.npy')) * 255).astype(np.uint8)
        frame = np.pad(frame[pad:-pad, pad:-pad], [[pad, pad], [pad, pad], [0, 0]], 'edge')
        self.model.apply(frame)

        frames = (np.load(os.path.join(FILE_PATH, '../dqn_image/scenes/rgb.npy')) * 255).astype(np.uint8)
        frames = np.pad(frames[:, pad:-pad, pad:-pad], [[0,0], [pad,pad], [pad,pad], [0,0]], 'edge')
        for frame in frames:
            self.model.apply(frame)

    def get_masks(self, image, n_cluster=3):
        pad = self.pad 
        #image = np.pad(image[pad:-pad, pad:-pad], [[pad,pad],[pad,pad], [0, 0]], 'edge').astype(np.uint8)
        fmask = self.model.apply(image, 0, 0)

        my, mx = np.nonzero(fmask)
        points = list(zip(mx, my, np.ones_like(mx) * 96))
        print(len(points))
        print()
        z = (np.array(points).T / np.linalg.norm(points, axis=1)).T

        im_blur = cv2.blur(image, (5, 5))
        colors = np.array([im_blur[y, x] / (10 * 255) for x, y in zip(mx, my)])
        z_color = np.concatenate([z, colors], 1)
        clusters = SpectralClustering(n_clusters=n_cluster, n_init=10).fit_predict(z_color)

        new_mask = np.zeros([fmask.shape[0], fmask.shape[1], n_cluster])
        for x, y, c in zip(mx, my, clusters):
            new_mask[y, x, c] = 1
        masks = new_mask.transpose([2,0,1]).astype(float)

        colors = []
        for mask in masks:
            color = image[mask.astype(bool)].mean(0) / 255.
            colors.append(color)

        return masks, np.array(colors), fmask

    def mask_over(self, image, threshold):
        return (image >= threshold).all(-1)

    def mask_under(self, image, threshold):
        return (image <= threshold).all(-1)

    def get_workspace_seg(self, image):
        pad = self.pad
        image = np.pad(image[pad:-pad, pad:-pad], [[pad,pad],[pad,pad], [0, 0]], 'edge')/255
        return self.mask_over(image, [.97, .97, .97])

    def make_empty_workspace_seg(self):
        pad = self.pad
        frame = (np.load(os.path.join(FILE_PATH, '../dqn_image/scenes/bg.npy')) * 255).astype(np.uint8)
        frame = np.pad(frame[pad:-pad, pad:-pad], [[pad, pad], [pad, pad], [0, 0]], 'edge')
        self.workspace_seg = self.get_workspace_seg(frame)


if __name__=='__main__':
    data_dir = '/ssd/disk/ur5_tidying_data/pybullet_single_bg2/images'
    img_path = '/ssd/disk/ur5_tidying_data/pybullet_single_bg2/images/00006.png'

    backsub = BackgroundSubtraction()
    backsub.fitting_model_from_data(data_dir, res=100)

    im = np.array(Image.open(img_path)) / 255.
    im = cv2.resize(im, (100, 100), interpolation=cv2.INTER_AREA)
    im = (im * 255).astype(np.uint8)
    #x = np.array(Image.open(img_path))[:, :, :3].astype(np.uint8)
    fmask = backsub.model.apply(im, 0, 0)
    plt.imshow(fmask)
    plt.show()

    masks, colors, fmask = backsub.get_masks(im, n_cluster=4)

    print(len(masks), 'objects detected.')
    for m in masks:
        plt.imshow(m)
        plt.show()

    print(fmask)
