import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from data_loader import TabletopTemplateDataset
from ellipse import LsqEllipse
from matplotlib.patches import Ellipse

dataDir = 'sample/'
removeBG = True #False
labelType = 'linspace'
view = 'top'
dataset = TabletopTemplateDataset(data_dir=dataDir, remove_bg=removeBG, label_type=labelType, view=view)

rgbList = sorted([os.path.join(p, 'rgb_%s.png'%view) for p in dataset.data_paths])
segList = sorted([os.path.join(p, 'seg_%s.npy'%view) for p in dataset.data_paths])

dataIndex = 2
segmap = np.load(segList[dataIndex])

masks, centers = [], []
for o in range(3):
    print()
    print(o)
    # get the segmentation mask of each object #
    mask = (segmap==o+2).astype(float)
    if mask.sum()<100:
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.dilate(cv2.erode(mask, kernel), kernel)
    else:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(cv2.erode(mask, kernel), kernel)
    mask = np.round(mask)
    masks.append(mask)
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
        plt.show()

        angle = phi * 180 / np.pi
        height, width = mask.shape[:2]
        matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        mask_rotated = cv2.warpAffine(mask, matrix, (width, height))

    center = (cy, cx)
    centers.append(center)
    plt.subplot(1, 2, 1)
    plt.imshow(mask)
    plt.subplot(1, 2, 2)
    plt.imshow(mask_rotated)
    plt.show()
