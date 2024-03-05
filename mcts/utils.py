import numpy as np
from ellipse import LsqEllipse

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

def loadRewardFunction(model_path):
    vNet = resnet18(pretrained=False)
    fc_in_features = vNet.fc.in_features
    vNet.fc = nn.Sequential(
        nn.Linear(fc_in_features, 1),
    )
    vNet.load_state_dict(torch.load(model_path))
    vNet.to("cuda:0")
    vNet.eval()
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return vNet, preprocess

class Renderer(object):
    def __init__(self, tableSize, imageSize, cropSize):
        self.tableSize = np.array(tableSize)
        self.imageSize = np.array(imageSize)
        self.cropSize = np.array(cropSize)
        self.rgb = None

    def setup(self, rgbImage, segmentation):
        # segmentation info
        # 1: background (table)
        # 2: None
        # 3: robot arm
        # 4~N: objects
        self.numObjects = int(np.max(segmentation)) - 3
        self.ratio, self.offset = self.getRatio()
        self.masks, self.centers = self.getMasks(segmentation)
        self.rgb = np.copy(rgbImage)
        oPatches, oMasks = self.getObjectPatches()
        self.objectPatches, self.objectMasks, self.objectAngles = self.getRotatedPatches(oPatches, oMasks)
        self.segmap = np.copy(segmentation)
        posMap = self.getTable(segmentation)
        rotMap = np.zeros_like(posMap)
        table = [posMap, rotMap]
        return table

    def getRatio(self):
        # v2.
        ratio = self.imageSize / self.tableSize
        offset = 0.0
        # ty, tx = np.round((np.array([py, px]) + 0.5) * ratio - 0.5).astype(int)
        # gy, gx = np.round((np.array(center) + 0.5) / ratio - 0.5).astype(int)

        # v1.
        # ratio = self.imageSize // self.tableSize
        # offset = (self.imageSize - ratio * self.tableSize + ratio)//2
        # ty, tx = np.array([py, px]) * self.ratio + self.offset
        # gy, gx = ((np.array(center) - self.offset) // self.ratio).astype(int)
        return ratio, offset
    
    def getMasks(self, segmap):
        masks, centers = [], []
        for o in range(self.numObjects):
            # get the segmentation mask of each object #
            mask = (segmap==o+4).astype(float)
            # if mask.sum()<100:
            #     kernel = np.ones((2, 2), np.uint8)
            #     mask = cv2.dilate(cv2.erode(mask, kernel), kernel)
            # else:
            #     kernel = np.ones((3, 3), np.uint8)
            #     mask = cv2.dilate(cv2.erode(mask, kernel), kernel)
            mask = np.round(mask)
            masks.append(mask)
            # get the center of each object #
            py, px = np.where(mask)
            cy, cx = np.round([np.mean(py), np.mean(px)])
            center = (cy, cx)
            centers.append(center)
        return masks, centers
    
    def getObjectPatches(self):
        objPatches = []
        objMasks = []
        for o in range(self.numObjects):
            mask = self.masks[o]
            cy, cx = self.centers[o]

            yMin = int(cy-self.cropSize[0]/2)
            yMax = int(cy+self.cropSize[0]/2)
            xMin = int(cx-self.cropSize[1]/2)
            xMax = int(cx+self.cropSize[1]/2)
            objPatch = np.zeros([*self.cropSize, 3])
            objPatch[
                max(0, -yMin): max(0, -yMin) + min(self.imageSize[0], yMax) - max(0, yMin),
                max(0, -xMin): max(0, -xMin) + min(self.imageSize[1], xMax) - max(0, xMin),
            ] = self.rgb[
                    max(0, yMin): min(self.imageSize[0], yMax),
                    max(0, xMin): min(self.imageSize[1], xMax),
                    :3
                ] * mask[
                        max(0, yMin): min(self.imageSize[0], yMax),
                        max(0, xMin): min(self.imageSize[1], xMax)
                    ][:, :, None]

            objMask = np.zeros(self.cropSize)
            objMask[
                max(0, -yMin): max(0, -yMin) + min(self.imageSize[0], yMax) - max(0, yMin),
                max(0, -xMin): max(0, -xMin) + min(self.imageSize[1], xMax) - max(0, xMin)
            ] = mask[
                    max(0, yMin): min(self.imageSize[0], yMax),
                    max(0, xMin): min(self.imageSize[1], xMax)
                ]
            # plt.imshow(objPatch/255.)
            # plt.show()
            objPatches.append(objPatch)
            objMasks.append(objMask)
        return objPatches, objMasks

    def getRotatedPatches(self, objPatches, objMasks, numRotations=2):
        rotatedObjPatches = [[] for _ in range(numRotations)]
        rotatedObjMasks = [[] for _ in range(numRotations)]
        rotatedAngles = [[] for _ in range(numRotations)]
        for o in range(len(objPatches)):
            patch = objPatches[o]
            mask = objMasks[o]
            py, px = np.where(mask)
            cy, cx = np.round([np.mean(py), np.mean(px)])
            X = np.array(list(zip(px, py)))
            if len(X) < 5:
                # can be a rectangle
                rect = cv2.minAreaRect(X)
                phi = rect[2]
            else:
                reg = LsqEllipse().fit(X)
                center, width, height, phi = reg.as_parameters()
                if np.abs(width-height) < 6:
                    # can be a rectangle
                    rect = cv2.minAreaRect(X)
                    phi = rect[2]
            for r in range(numRotations):
                angle = phi / np.pi * 180 + r * 90
                height, width = mask.shape[:2]
                matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                patch_rotated = cv2.warpAffine(patch, matrix, (width, height))
                mask_rotated = cv2.warpAffine(mask, matrix, (width, height))
                rotatedObjPatches[r].append(patch_rotated)
                rotatedObjMasks[r].append(mask_rotated)
                rotatedAngles[r].append(angle)

        objectPatches = [objPatches] + rotatedObjPatches
        objectMasks = [objMasks] + rotatedObjMasks
        objectAngles = [[0. for _ in range(len(objPatches))]] + rotatedAngles
        if False: # check patches
            for r in range(len(objectPatches)):
                for o in range(len(objectPatches[r])):
                    plt.imshow(objectPatches[r][o]/255.)
                    plt.savefig('data/mcts-ellipse/%d_%d.png'%(o, r))
        return objectPatches, objectMasks, objectAngles

    def getRGB(self, table, remove=None):
        posMap, rotMap = table
        newRgb = np.zeros_like(np.array(self.rgb))[:, :, :3]
        for o in range(self.numObjects):
            if remove is not None:
                if o==remove:
                    continue
            if (posMap==o+1).any():
                py, px = np.where(posMap==o+1)
                py, px = py[0], px[0]
                ty, tx = np.round((np.array([py, px]) + 0.5) * self.ratio - 0.5).astype(int)
                # ty, tx = np.array([py, px]) * self.ratio + self.offset
                rot = int(rotMap[py, px])
            else:
                ty, tx = self.centers[o]
                rot = 0
            yMin = int(ty - self.cropSize[0] / 2)
            yMax = int(ty + self.cropSize[0] / 2)
            xMin = int(tx - self.cropSize[1] / 2)
            xMax = int(tx + self.cropSize[1] / 2)
            newRgb[
                max(0, yMin): min(self.imageSize[0], yMax),
                max(0, xMin): min(self.imageSize[1], xMax)
            ] += (self.objectPatches[rot][o] * self.objectMasks[rot][o][:, :, None])[
                    max(0, -yMin): max(0, -yMin) + (min(self.imageSize[0], yMax) - max(0, yMin)),
                    max(0, -xMin): max(0, -xMin) + (min(self.imageSize[1], xMax) - max(0, xMin)),
                ].astype(np.uint8)
        # plt.imshow(newRgb)
        # plt.show()
        return np.array(newRgb)

    def getTable(self, segmap):
        newTable = np.zeros([self.tableSize[0], self.tableSize[1]])
        # return newTable
        for o in range(self.numObjects):
            center = self.centers[o]
            gyx = (np.array(center) + 0.5) / self.ratio - 0.5
            if np.linalg.norm(gyx - np.round(gyx))<0.2:
                gy, gx = np.round(gyx).astype(int)
                # gy, gx = np.round((np.array(center) + 0.5) / self.ratio - 0.5).astype(int)
                # gy, gx = ((np.array(center) - self.offset) // self.ratio).astype(int)
                newTable[gy, gx] = o + 1
        return newTable

    def convert_action(self, action):
        obj, py, px, rot = action
        target_object = obj + 3
        ty, tx = np.round((np.array([py, px]) + 0.5) * self.ratio - 0.5).astype(int)
        # ty, tx = np.array([py, px]) * self.ratio + self.offset
        target_position = [ty, tx]

        rot_angle = self.objectAngles[rot][obj-1]
        rot_angle = rot_angle / 180 * np.pi
        return target_object, target_position, rot_angle
