import os
import sys
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '..'))
from data_loader import TabletopTemplateDataset
from ellipse import LsqEllipse
from matplotlib.patches import Ellipse

dataDir = '/ssd/disk/TableTidyingUp/dataset_template/train'
removeBG = True #False
labelType = 'linspace'
view = 'top'
dataset = TabletopTemplateDataset(data_dir=dataDir, remove_bg=removeBG, label_type=labelType, view=view)

rgbList = sorted([os.path.join(p, 'rgb_%s.png'%view) for p in dataset.data_paths])
segList = sorted([os.path.join(p, 'seg_%s.npy'%view) for p in dataset.data_paths])

count = 0
for dataIndex in np.random.choice(len(segList), 10):
    segmap = np.load(segList[dataIndex])
    rgb = np.array(Image.open(rgbList[dataIndex]))

    masks, centers = [], []
    for o in range(3):
        print()
        print(o)
        # get the segmentation mask of each object #
        mask = (segmap==o+2).astype(float)
        mask = np.round(mask)
        masks.append(mask)
        rgb_masked = rgb[:, :, :3] * mask[:, :, None] / 255.
        # get the center of each object #
        py, px = np.where(mask)
        cy, cx = np.round([np.mean(py), np.mean(px)])

        if True:
            X = np.array(list(zip(px, py)))
            reg = LsqEllipse().fit(X)
            center, width, height, phi = reg.as_parameters()
            print(f'center: {center[0]:.3f}, {center[1]:.3f}')
            print(f'width: {width:.3f}')
            print(f'height: {height:.3f}')
            print(f'phi: {phi:.3f}')

            fig = plt.figure(figsize=(6, 6))
            ax = plt.subplot()
            ax.axis('equal')
            ax.plot(px, py, 'ro', zorder=1)
            ellipse = Ellipse(
                xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
                edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
            )
            ax.add_patch(ellipse)

            plt.xlabel('$X_1$')
            plt.ylabel('$X_2$')

            plt.legend()
            plt.savefig('../data/weekly/ellipse_%d.png'%count)
            #plt.show()

            angle = phi * 180 / np.pi
            height, width = mask.shape[:2]
            matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            mask_rotated = cv2.warpAffine(mask, matrix, (width, height))
            rgb_rotated = cv2.warpAffine(rgb_masked, matrix, (width, height))

            angle2 = phi * 180 / np.pi + 90
            height, width = mask.shape[:2]
            matrix2 = cv2.getRotationMatrix2D((cx, cy), angle2, 1.0)
            mask_rotated2 = cv2.warpAffine(mask, matrix2, (width, height))
            rgb_rotated2 = cv2.warpAffine(rgb_masked, matrix2, (width, height))

        center = (cy, cx)
        centers.append(center)
        figure = plt.figure(figsize=(10, 5))
        plt.subplot(2, 3, 1)
        plt.imshow(mask)
        plt.subplot(2, 3, 2)
        plt.imshow(mask_rotated)
        plt.subplot(2, 3, 3)
        plt.imshow(mask_rotated2)
        plt.subplot(2, 3, 4)
        plt.imshow(rgb_masked)
        plt.subplot(2, 3, 5)
        plt.imshow(rgb_rotated)
        plt.subplot(2, 3, 6)
        plt.imshow(rgb_rotated2)
        plt.savefig('../data/weekly/rotated_patch_%d.png'%count)
        #plt.show()
        count += 1
