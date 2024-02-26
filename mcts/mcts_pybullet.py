import os
import sys
import copy
import cv2
import datetime
import time
import random
import numpy as np
import logging
import json
import pybullet as p
from argparse import ArgumentParser
from PIL import Image
from matplotlib import pyplot as plt
from ellipse import LsqEllipse

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from data_loader import TabletopTemplateDataset

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp/pybullet_ur5_robotiq'))
from custom_env import TableTopTidyingUpEnv, get_contact_objects
from utilities import Camera, Camera_front_top


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
        ratio = self.imageSize // self.tableSize
        offset = (self.imageSize - ratio * self.tableSize + ratio)//2
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
        for o in range(1, self.numObjects+1):
            if remove is not None:
                if o==remove:
                    continue
            if (posMap==o).any():
                py, px = np.where(posMap==o)
                py, px = py[0], px[0]
                ty, tx = np.array([py, px]) * self.ratio + self.offset
                rot = int(rotMap[py, px])
            else:
                ty, tx = self.centers[o-1]
                rot = 0
            yMin = int(ty - self.cropSize[0] / 2)
            yMax = int(ty + self.cropSize[0] / 2)
            xMin = int(tx - self.cropSize[1] / 2)
            xMax = int(tx + self.cropSize[1] / 2)
            newRgb[
                max(0, yMin): min(self.imageSize[0], yMax),
                max(0, xMin): min(self.imageSize[1], xMax)
            ] += (self.objectPatches[rot][o-1] * self.objectMasks[rot][o-1][:, :, None])[
                    max(0, -yMin): max(0, -yMin) + (min(self.imageSize[0], yMax) - max(0, yMin)),
                    max(0, -xMin): max(0, -xMin) + (min(self.imageSize[1], xMax) - max(0, xMin)),
                ].astype(np.uint8)
        # plt.imshow(newRgb)
        # plt.show()
        return np.array(newRgb)

    def getTable(self, segmap):
        newTable = np.zeros([self.tableSize[0], self.tableSize[1]])
        return newTable

    def convert_action(self, action):
        obj, py, px, rot = action
        target_object = obj + 3
        ty, tx = np.array([py, px]) * self.ratio + self.offset
        target_position = [ty, tx]

        rot_angle = self.objectAngles[rot][obj-1]
        rot_angle = rot_angle / 180 * np.pi
        return target_object, target_position, rot_angle

        
class Node(object):
    def __init__(self, numObjects, table, parent=None):
        self.table = table
        self.parent = parent
        self.numObjcts = numObjects
        if parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

        #self.isTerminal = False
        self.numVisits = 0
        self.totalReward = 0.
        self.children = {}
        self.actionCandidates = []
    
    def takeAction(self, move):
        obj, py, px, rot = move
        posMap, rotMap = self.table
        newPosMap = copy.deepcopy(posMap)
        newRotMap = copy.deepcopy(rotMap)
        newPosMap[posMap==obj] = 0
        newPosMap[py, px] = obj
        newRotMap[posMap==obj] = 0
        newRotMap[py, px] = rot
        newTable = [newPosMap, newRotMap]
        return newTable
    
    def setActions(self, actionCandidates):
        self.actionCandidates = actionCandidates

    def isFullyExpanded(self):
        return len(self.children)!=0 and len(self.children)==len(self.actionCandidates)

    def __str__(self):
        s=[]
        s.append("Reward: %s"%(self.totalReward))
        s.append("Visits: %d"%(self.numVisits))
        #s.append("Terminal: %s"%(self.isTerminal))
        s.append("Children: %d"%(len(self.children.keys())))
        return "%s: %s"%(self.__class__.__name__, ' / '.join(s))


class MCTS(object):
    def __init__(self, renderer, args, explorationConstant=1/np.sqrt(2)):
        timeLimit = args.time_limit
        iterationLimit = args.iteration_limit
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.renderer = renderer
        if self.renderer is None:
            self.domain = 'grid'
        else:
            self.domain = 'image'
        self.explorationConstant = explorationConstant

        self.treePolicy = args.tree_policy
        self.maxDepth = args.max_depth
        rolloutPolicy = args.rollout_policy
        if rolloutPolicy=='random':
            self.rollout = self.randomPolicy
        elif rolloutPolicy=='nostep':
            self.rollout = self.noStepPolicy
        elif rolloutPolicy=='onestep':
            self.rollout = self.oneStepPolicy
        else:
            self.rollout = lambda n: self.greedyPolicy(n, rolloutPolicy)

        self.thresholdSuccess = args.threshold_success #0.6
        self.thresholdQ = args.threshold_q #0.5
        self.thresholdV = args.threshold_v #0.5
        self.thresholdProb = args.threshold_prob #0.1
        self.QNet = None
        self.VNet = None
        self.policyNet = None
        self.batchSize = args.batch_size #32
        self.preProcess = None
    
    def reset(self, rgbImage, segmentation):
        table = self.renderer.setup(rgbImage, segmentation)
        return table

    def setQNet(self, QNet):
        self.QNet = QNet
    
    def setVNet(self, VNet):
        self.VNet = VNet

    def setPolicyNet(self, policyNet):
        self.policyNet = policyNet

    def setPreProcess(self, preProcess):
        self.preProcess = preProcess

    def search(self, table, needDetails=False):
        # print('search.')
        self.root = Node(self.renderer.numObjects, table)
        if table is not None:
            self.root = Node(self.renderer.numObjects, table)
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()
        bestChild = self.getBestChild(self.root, explorationValue=0.)
        action=(action for action, node in self.root.children.items() if node is bestChild).__next__()
        if needDetails:
            return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        else:
            return action

    def selectNode(self, node):
        # print('selectNode.')
        while not self.isTerminal(node)[0]:
            if len(node.children)==0:
                return self.expand(node)
            elif random.uniform(0, 1) < 0.5:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                if node.isFullyExpanded():
                    # print('fully expanded.')
                    # print('nodes:', len(node.children))
                    node = self.getBestChild(node, self.explorationConstant)
                else:
                    return self.expand(node)
        return node

    def expand(self, node):
        # print('expand.')
        while True:
            actions = self.getPossibleActions(node, self.treePolicy)
            assert actions is not None
            action = random.choice(actions)
            if tuple(action) not in node.children:
                newNode = Node(self.renderer.numObjects, node.takeAction(action), node)
                node.children[tuple(action)] = newNode
                return newNode

    def getReward(self, table):
        # print('getReward.')
        rgb = self.renderer.getRGB(table)
        s = torch.Tensor(rgb[None, :]/255.).permute([0,3,1,2]).cuda()
        if self.preProcess is not None:
            s = self.preProcess(s)
        reward = self.VNet(s).cpu().detach().numpy()[0]
        #reward = max(0, (reward - 0.5) * 2)
        return reward

    def backpropogate(self, node, reward):
        # print('backpropagate.')
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def executeRound(self):
        # print('executeRound.')
        node = self.selectNode(self.root)
        reward = self.rollout(node)
        self.backpropogate(node, reward)

    def getBestChild(self, node, explorationValue):
        # print('getBestChild.')
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.totalReward / child.numVisits + explorationValue * \
                    np.sqrt(2 * np.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return np.random.choice(bestNodes)

    def getPossibleActions(self, node, policy='random', needValues=False):
        # print('getPossibleActions.')
        if policy=='Q':
            if len(node.actionCandidates)==0:
                actionCandidates = []
                states = []
                objectPatches = []
                for o in range(len(self.renderer.numObjects)):
                    rgbWoTarget = self.renderer.getRGB(node.table, remove=o)
                    objPatch = self.renderer.objectPatches[o]
                    states.append(rgbWoTarget)
                    objectPatches.append(objPatch)
                s = torch.Tensor(np.array(states)/255.).permute([0,3,1,2]).cuda()
                p = torch.Tensor(np.array(objectPatches)/255.).permute([0,3,1,2]).cuda()
                if self.preProcess is not None:
                    s = self.preProcess(s)
                    p = self.reProcess(p)
                QHeatmap = self.QNet(s, p).cpu().detach().numpy()
                for o, py, px in np.where(QHeatmap > self.thresholdQ):
                    actionCandidates.append((o, py, px))
                node.setActions(actionCandidates)
            return node.actionCandidates

        elif policy=='V':
            if len(node.actionCandidates)==0:
                nb = self.renderer.numObjects
                th, tw = self.renderer.tableSize
                allPossibleActions = np.array(np.meshgrid(
                                np.arange(1, nb+1), np.arange(th), np.arange(tw), np.arange(1,3)
                                )).T.reshape(-1, 4)
                nextStates = []
                for action in allPossibleActions:
                    nextState = self.renderer.getRGB(node.takeAction(action))
                    nextStates.append(nextState)
                s = torch.Tensor(np.array(nextStates)/255.).permute([0,3,1,2]).cuda()
                if self.preProcess is not None:
                    s = self.preProcess(s)
                values = self.VNet(s).cpu().detach().numpy()
                possibleIdx = values > self.thresholdV
                node.setActions(allPossibleActions[possibleIdx])
            if needValues:
                return node,actionCandidates, values[possibleIdx]
            else:
                return node.actionCandidates
            
        elif policy=='policy':
            if len(node.actionCandidates)==0:
                actionCandidates = []
                states = []
                objectPatches = []
                for o in range(len(self.renderer.numObjects)):
                    rgbWoTarget = self.renderer.getRGB(node.table, remove=o)
                    states.append(rgbWoTarget)
                s = torch.Tensor(np.array(states)/255.).permute([0,3,1,2]).cuda()
                if self.preProcess is not None:
                    s = self.preProcess(s)
                probMap  = self.policyNet(s).cpu().detach().numpy()
                for o, py, px in np.where(probMap > self.thresholdProb):
                    actionCandidates.append((o, py, px))
                node.setActions(actionCandidates)
            return node.actionCandidates

        elif policy=='random':
            if len(node.actionCandidates)==0:
                nb = self.renderer.numObjects
                th, tw = self.renderer.tableSize
                allPossibleActions = np.array(np.meshgrid(
                                np.arange(1, nb+1), np.arange(th), np.arange(tw), np.arange(1,3)
                                )).T.reshape(-1, 4)
                allPossibleActions = [a for a in allPossibleActions if self.root.table[0][a[1], a[2]]==0]
                node.setActions(allPossibleActions)
            return node.actionCandidates

    def isTerminal(self, node, table=None, checkReward=False):
        # print('isTerminal')
        if table is None:
            if node.depth >= self.maxDepth:
                return True, 0.0
            table = node.table
        #for o in range(1, self.renderer.numObjects+1):
        #    if len(np.where(table==o)[0])==0:
        #        return True, 0.0
        if checkReward:
            reward = self.getReward(table)
            if reward > self.thresholdSuccess:
                return True, reward
            else:
                return False, reward
        return False, 0.0

    def noStepPolicy(self, node):
        # print('noStepPolicy.')
        # st = time.time()
        states = [self.renderer.getRGB(node.table)]
        s = torch.Tensor(np.array(states)/255.).permute([0,3,1,2]).cuda()
        if self.preProcess is not None:
            s = self.preProcess(s)
        rewards = self.VNet(s).cpu().detach().numpy()
        #rewards = np.maximum(0, (rewards - 0.5) * 2)
        maxReward = np.max(rewards)
        # et = time.time()
        # print(et - st, 'seconds.')
        return maxReward

    def oneStepPolicy(self, node):
        # print('oneStepPolicy.')
        # st = time.time()
        nb = self.renderer.numObjects
        th, tw = self.renderer.tableSize
        allPossibleActions = np.array(np.meshgrid(
                        np.arange(1, nb+1), np.arange(th), np.arange(tw), np.arange(1,3)
                        )).T.reshape(-1, 4)
        states = [self.renderer.getRGB(node.table)]
        for action in allPossibleActions:
            newTable = node.takeAction(action)
            newNode = Node(self.renderer.numObjects, newTable)
            states.append(self.renderer.getRGB(newNode.table))

        s = torch.Tensor(np.array(states)/255.).permute([0,3,1,2]).cuda()
        if self.preProcess is not None:
            s = self.preProcess(s)
        rewards = []
        numBatches = len(states)//self.batchSize
        if len(states)%self.batchSize > 0:
            numBatches += 1
        for b in range(numBatches):
            batchS = s[b*self.batchSize:(b+1)*self.batchSize]
            batchRewards = self.VNet(batchS).cpu().detach().numpy()
            rewards.append(batchRewards)
        rewards = np.concatenate(rewards)
        maxReward = np.max(rewards)
        # et = time.time()
        # print(et - st, 'seconds.')
        return maxReward

    def randomPolicy(self, node):
        # print('randomPolicy.')
        # st = time.time()
        nb = self.renderer.numObjects
        th, tw = self.renderer.tableSize
        allPossibleActions = np.array(np.meshgrid(
                            np.arange(1, nb+1), np.arange(th), np.arange(tw), np.arange(1,3)
                            )).T.reshape(-1, 4)
        states = [self.renderer.getRGB(node.table)]
        while not self.isTerminal(node)[0]:
            try:
                action = random.choice(allPossibleActions)
            except IndexError:
                raise Exception("Non-terminal state has no possible actions: " + str(state))
            newTable = node.takeAction(action)
            newNode = Node(self.renderer.numObjects, newTable)
            node = newNode
            states.append(self.renderer.getRGB(node.table))
        s = torch.Tensor(np.array(states)/255.).permute([0,3,1,2]).cuda()
        if self.preProcess is not None:
            s = self.preProcess(s)
        rewards = []
        numBatches = len(states)//self.batchSize
        if len(states)%self.batchSize > 0:
            numBatches += 1
        for b in range(numBatches):
            batchS = s[b*self.batchSize:(b+1)*self.batchSize]
            batchRewards = self.VNet(batchS).cpu().detach().numpy()
            rewards.append(batchRewards)
        rewards = np.concatenate(rewards)
        #rewards = self.VNet(s).cpu().detach().numpy()
        maxReward = np.max(rewards)
        # et = time.time()
        # print(et - st, 'seconds.')
        return maxReward

    def greedyPolicyWithQ(self, node):
        states = [self.renderer.getRGB(node.table)]
        while not self.isTerminal(node)[0]:
            try:
                actions = self.getPossibleActions(node, 'Q')
                action = np.random.choice(actions)
            except IndexError:
                raise Exception("Non-terminal state has no possible actions: " + str(state))
            newTable = node.takeAction(action)
            newNode = Node(self.renderer.numObjects, newTable)
            node = newNode
            states.append(self.renderer.getRGB(node.table))
        s = torch.Tensor(np.array(states)/255.).permute([0,3,1,2]).cuda()
        if self.preProcess is not None:
            s = self.preProcess(s)
        rewards = self.VNet(s).cpu().detach().numpy()
        maxReward = np.max(rewards)
        return maxReward

    def greedyPolicyWithV(self, node):
        state = self.renderer.getRGB(node.table)
        s = torch.Tensor(state/255.).permute([0,3,1,2]).cuda()
        if self.preProcess is not None:
            s = self.preProcess(s)
        value = self.VNet(s).cpu().detach().numpy()[0]
        values = [value]
        while not self.isTerminal(node)[0]:
            try:
                actions, values = self.getPossibleActions(node, 'V', needValues=True)
                idx = np.random.choice(len(actions))
                action, value = actions[idx], values[idx]
            except IndexError:
                raise Exception("Non-terminal state has no possible actions: " + str(state))
            newTable = node.takeAction(action)
            newNode = Node(self.renderer.numObjects, newTable)
            node = newNode
            values.append(value)
        maxReward = np.max(values)
        return maxReward


def setupEnvironment(objects, args):
    camera_top = Camera((0, 0, 1.45), 0.02, 2, (480, 360), 60)
    camera_front_top = Camera_front_top((0.5, 0, 1.3), 0.02, 2, (480, 360), 60)
    
    objects_cfg = { 'paths': {
            'pybullet_object_path' : '/ssd/disk/pybullet-URDF-models/urdf_models/models',
            'ycb_object_path' : '/ssd/disk/YCB_dataset',
            'housecat_object_path' : '/ssd/disk/housecat6d/obj_models_small_size_final',
        },
        'split' : 'inference' #'train'
    }
    
    gui_on = not args.gui_off
    env = TableTopTidyingUpEnv(objects_cfg, camera_top, camera_front_top, vis=gui_on, num_objs=args.num_objects, gripper_type='85')
    p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off
    p.addUserDebugLine([0, -0.5, 0], [0, -0.5, 1.1], [0, 1, 0])

    env.reset()
    object_pybullet_ids = env.spawn_objects(objects)
    
    env.arrange_objects(random=True)
    return env


if __name__=='__main__':
    parser = ArgumentParser()
    # Inference
    parser.add_argument('--num-objects', type=int, default=4)
    parser.add_argument('--num-scenes', type=int, default=10)
    parser.add_argument('--H', type=int, default=12)
    parser.add_argument('--W', type=int, default=15)
    parser.add_argument('--crop-size', type=int, default=96)
    parser.add_argument('--gui-off', action="store_true")
    # MCTS
    parser.add_argument('--time-limit', type=int, default=None)
    parser.add_argument('--iteration-limit', type=int, default=10000)
    parser.add_argument('--max-depth', type=int, default=7)
    parser.add_argument('--rollout-policy', type=str, default='nostep')
    parser.add_argument('--tree-policy', type=str, default='random')
    parser.add_argument('--threshold-success', type=float, default=0.85)
    parser.add_argument('--threshold-q', type=float, default=0.5)
    parser.add_argument('--threshold-v', type=float, default=0.5)
    parser.add_argument('--threshold-prob', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    # Reward model
    parser.add_argument('--reward-model-path', type=str, default='data/classification-best/top_nobg_linspace_mse-best.pth')
    parser.add_argument('--label-type', type=str, default='linspace')
    parser.add_argument('--view', type=str, default='top') 
    # Pretrained Models
    parser.add_argument('--qnet-path', type=str, default='')
    parser.add_argument('--vnet-path', type=str, default='')
    parser.add_argument('--policynet-path', type=str, default='../policy_learning/logs/0224_1815/pnet_e30.pth')
    args = parser.parse_args()

    # Logger
    now = datetime.datetime.now()
    log_name = now.strftime("%m%d_%H%M")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Environment setup
    objects = [('bowl', 'medium'), ('can_drink','medium'), ('plate','medium'), ('marker', 'medium'), ('soap_dish', 'medium'), ('book', 'medium'), ('remote', 'medium')]
    selected_objects = [objects[i] for i in np.random.choice(len(objects), args.num_objects, replace=False)]
    env = setupEnvironment(selected_objects, args)

    # MCTS setup
    renderer = Renderer(tableSize=(args.H, args.W), imageSize=(360, 480), cropSize=(args.crop_size, args.crop_size))
    searcher = MCTS(renderer, args)

    # Network setup
    model_path = args.reward_model_path
    vNet, preprocess = loadRewardFunction(model_path)
    searcher.setVNet(vNet)
    searcher.setPreProcess(preprocess)

    # Policy-based MCTS
    if args.tree_policy=='policy':
        sys.path.append(os.path.join(FILE_PATH, '..', 'policy_learning'))
        from model import ResNet
        pnet = ResNet()
        pnet.load_state_dict(torch.load(args.policynet_path))
        pnet = pnet.to("cuda:0")
        MCTS.setPolicyNet(pnet)
    
    for sidx in range(args.num_scenes):
        # setup logger
        os.makedirs('data/mcts-%s/scene-%d'%(log_name, sidx), exist_ok=True)
        with open('data/mcts-%s/config.json'%log_name, 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -\n%(message)s')
        file_handler = logging.FileHandler('data/mcts-%s/scene-%d/mcts.log'%(log_name, sidx))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Initial state
        obs = env.reset()
        selected_objects = [objects[i] for i in np.random.choice(len(objects), args.num_objects, replace=False)]
        env.spawn_objects(selected_objects)
        while True:
            is_occluded = False
            is_collision = False
            env.arrange_objects(random=True)
            obs = env.get_observation()
            initRgb = obs[args.view]['rgb']
            initSeg = obs[args.view]['segmentation']
            # Check occlusions
            for o in range(args.num_objects):
                # get the segmentation mask of each object #
                mask = (initSeg==o+4).astype(float)
                if mask.sum()==0:
                    print("Object %d is occluded."%o)
                    logger.info("Object %d is occluded."%o)
                    is_occluded = True
                    break
            # Check collision
            contact_objects = get_contact_objects()
            contact_objects = [c for c in list(get_contact_objects()) if 1 not in c and 2 not in c]
            if len(contact_objects) > 0:
                print("Collision detected.")
                print(contact_objects)
                logger.info("Collision detected.")
                logger.info(contact_objects)
                is_collision = True
            if is_occluded or is_collision:
                continue
            else:
                break

        plt.imshow(initRgb)
        plt.savefig('data/mcts-%s/scene-%d/initial.png'%(log_name, sidx))
        initTable = searcher.reset(initRgb, initSeg)
        print(initTable[0])
        logger.info('initTable: %s' % initTable)
        table = initTable

        print("--------------------------------")
        logger.info('-'*50)
        for step in range(10):
            resultDict = searcher.search(table=table, needDetails=True)
            print("Num Children: %d"%len(searcher.root.children))
            logger.info("Num Children: %d"%len(searcher.root.children))
            for i, c in enumerate(searcher.root.children):
                print(i, c, searcher.root.children[c])
            action = resultDict['action']

            # expected result in mcts #
            nextTable = searcher.root.takeAction(action)
            print("Best Action:", action)
            print("Best Child: \n %s"%nextTable[0])
            logger.info("Best Action: %s"%str(action))
            logger.info("Best Child: \n %s"%nextTable[0])
            tableRgb = renderer.getRGB(nextTable)
            plt.imshow(tableRgb)
            plt.savefig('data/mcts-%s/scene-%d/expect_%d.png'%(log_name, sidx, step))
            #plt.show()

            # simulation step in pybullet #
            target_object, target_position, rot_angle = renderer.convert_action(action)
            obs = env.step(target_object, target_position, rot_angle)
            currentRgb = obs[args.view]['rgb']
            currentSeg = obs[args.view]['segmentation']
            table = searcher.reset(currentRgb, currentSeg)
            #table = copy.deepcopy(nextTable)
            print("Current state: \n %s"%table[0])
            logger.info("Current state: \n %s"%table[0])
            plt.imshow(currentRgb)
            plt.savefig('data/mcts-%s/scene-%d/real_%d.png'%(log_name, sidx, step))
            terminal, reward = searcher.isTerminal(None, table, checkReward=True)
            print("Current Score:", reward)
            print("--------------------------------")
            logger.info("Current Score: %f" %reward)
            logger.info("-"*50)
            if terminal:
                print("Arrived at the final state:")
                print("Score:", reward)
                print(table[0])
                print("--------------------------------")
                print("--------------------------------")
                logger.info("Arrived at the final state:")
                logger.info("Score: %f"%reward)
                logger.info(table[0])
                plt.imshow(currentRgb)
                plt.savefig('data/mcts-%s/scene-%d/final.png'%(log_name, sidx))
                # plt.show()
                break
