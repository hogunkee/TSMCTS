import os
import cv2
import numpy as np

from sklearn.cluster import SpectralClustering

class BackgroundSubtraction():
    def __init__(self, pad=4):
        self.pad = pad
        self.model = None
        #self.fitting_model()

    def fitting_model(self, file_path):
        pad = self.pad 
        self.model = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        frames = (np.load(file_path) * 255).astype(np.uint8)
        frames = np.pad(frames[:, pad:-pad, pad:-pad], [[0,0], [pad,pad], [pad,pad], [0,0]], 'edge')
        for frame in frames:
            self.model.apply(frame)

    def get_masks(self, image, n_cluster=3):
        pad = self.pad 
        image = np.pad(image[pad:-pad, pad:-pad], [[pad,pad],[pad,pad], [0, 0]], 'edge').astype(np.uint8)
        fmask = self.model.apply(image, 0, 0)

        my, mx = np.nonzero(fmask)
        points = list(zip(mx, my, np.ones_like(mx) * 96))
        z = (np.array(points).T / np.linalg.norm(points, axis=1)).T

        im_blur = cv2.blur(image, (5, 5))
        colors = np.array([im_blur[y, x] / (10 * 255) for x, y in zip(mx, my)])
        z_color = np.concatenate([z, colors], 1)
        clusters = SpectralClustering(n_clusters=n_cluster, n_init=10).fit_predict(z_color)

        new_mask = np.zeros([fmask.shape[0], fmask.shape[1], n_cluster])
        for x, y, c in zip(mx, my, clusters):
            new_mask[y, x, c] = 1
        masks = new_mask.transpose([2,0,1]).astype(float)

        # # Opening
        # for i in range(len(masks)):
        #     masks[i] = cv2.morphologyEx(masks[i], cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        colors = []
        for mask in masks:
            color = image[mask.astype(bool)].mean(0) / 255.
            colors.append(color)

        return masks, np.array(colors), fmask

        # contours, hierarchy = cv2.findContours(fmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # masks, colors = [], []
        # num_seg = len(contours)
        # for ns in range(num_seg):
        #     zeros = np.zeros_like(fmask)
        #     obj_mask = cv2.drawContours(zeros, contours, ns, 1, -1)
        #     obj_color = image[obj_mask.astype(bool)].mean(0)/255.
        #     masks.append(obj_mask)
        #     colors.append(obj_color)
        # return np.array(masks), np.array(colors), fmask, contours

