import os
import copy
import cv2
import time
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn


class Renderer(object):
    def __init__(self, tableSize, imageSize, cropSize):
        self.tableSize = np.array(tableSize)
        self.imageSize = np.array(imageSize)
        self.cropSize = np.array(cropSize)
        self.rgb = None

    def setup(self, rgbImage, segmentation):
        self.numObjects = int(np.max(segmentation))
        self.ratio, self.offset = self.getRatio()
        self.masks, self.centers = self.getMasks(segmentation)
        self.rgb = np.copy(rgbImage)
        self.objectPatches, self.objectMasks = self.getObjectPatches()
        self.segmap = np.copy(segmentation)
        table = self.getTable(segmentation)
        return table

    def getRatio(self):
        ratio = self.imageSize // self.tableSize
        offset = (self.imageSize - ratio * self.tableSize + ratio)//2
        return ratio, offset
    
    def getMasks(self, segmap):
        masks, centers = [], []
        for o in range(1, self.numObjects+1):
            # get the segmentation mask of each object #
            mask = (segmap==o).astype(float)
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
            center = (cy, cx)
            centers.append(center)
        return masks, centers
    
    def getObjectPatches(self):
        objPatches = []
        objMasks = []
        for o in range(self.numObjects):
            mask = self.masks[o]
            cy, cx = self.centers[o]
            objPatch = np.zeros([*self.cropSize, 3])
            objPatch[:, :] = self.rgb[ 
                        int(cy-self.cropSize[0]/2):int(cy+self.cropSize[0]/2), 
                        int(cx-self.cropSize[1]/2):int(cx+self.cropSize[1]/2),
                        :3
                        ]
            objMask = np.zeros(self.cropSize)
            objMask[:, :] = mask[ 
                        int(cy-self.cropSize[0]/2):int(cy+self.cropSize[0]/2), 
                        int(cx-self.cropSize[1]/2):int(cx+self.cropSize[1]/2)
                        ]
            objPatches.append(objPatch)
            objMasks.append(objMask)
        return objPatches, objMasks

    def getRGB(self, table, remove=None):
        newRgb = np.zeros_like(np.array(self.rgb))[:, :, :3]
        for o in range(1, self.numObjects+1):
            if remove is not None:
                if o==remove:
                    continue
            py, px = np.where(table==o)
            if len(py)==0 or len(px)==0:
                continue
            py, px = py[0], px[0]
            ty, tx = np.array([py, px]) * self.ratio + self.offset
            newRgb[
                    int(ty-self.cropSize[0]/2):int(ty+self.cropSize[0]/2),
                    int(tx-self.cropSize[1]/2):int(tx+self.cropSize[1]/2)
                    ] += (self.objectPatches[o-1] * self.objectMasks[o-1][:, :, None]).astype(np.uint8)
        return np.array(newRgb)

    def getTable(self, segmap):
        newTable = np.zeros([self.tableSize[0], self.tableSize[1]])
        for o in range(1, self.numObjects+1):
            center = self.centers[o-1]
            gy, gx = ((np.array(center) - self.offset) // self.ratio).astype(int)
            newTable[gy, gx] = o
        return newTable

    def convertAction(self, moveIdx):
        obj = moveIdx // (self.tableSize[0]*self.tableSize[1]) + 1
        py = (moveIdx % (self.tableSize[0]*self.tableSize[1])) // self.tableSize[1] 
        px = (moveIdx % (self.tableSize[0]*self.tableSize[1])) % self.tableSize[0] 
        return (obj, py, px)
    
        
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
        obj, py, px = move #self.convert_action(move)
        newTable = copy.deepcopy(self.table)
        newTable[newTable==obj] = 0
        newTable[py, px] = obj
        return newTable
    
    def setActions(self, actionCandidates):
        self.actionCandidates = actionCandidates

    def isFullyExpanded(self):
        return len(self.children)!=0 and (self.children)==len(self.actionCandidates)

    def __str__(self):
        s=[]
        s.append("Reward: %s"%(self.totalReward))
        s.append("Visits: %d"%(self.numVisits))
        #s.append("Terminal: %s"%(self.isTerminal))
        s.append("Children: %d"%(len(self.children.keys())))
        return "%s: %s"%(self.__class__.__name__, ' / '.join(s))


class MCTS(object):
    def __init__(self, timeLimit=None, iterationLimit=None, renderer=None, 
                 explorationConstant=1/np.sqrt(2), maxDepth=8,
                 rolloutPolicy='random', treePolicy='random'):
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

        self.treePolicy = treePolicy
        self.maxDepth = maxDepth
        if rolloutPolicy=='random':
            self.rollout = self.randomPolicy
        elif rolloutPolicy=='nostep':
            self.rollout = self.noStepPolicy
        elif rolloutPolicy=='onestep':
            self.rollout = self.oneStepPolicy
        else:
            self.rollout = lambda n: self.greedyPolicy(n, rolloutPolicy)

        self.thresholdSuccess = 0.9
        self.thresholdQ = 0.5
        self.thresholdV = 0.5
        self.QNet = None
        self.VNet = None
        self.batchSize = 32
    
    def reset(self, rgbImage, segmentation):
        table = self.renderer.setup(rgbImage, segmentation)
        return table

    def setQNet(self, QNet):
        self.QNet = QNet
    
    def setVNet(self, VNet):
        self.VNet = VNet

    def search(self, table, needDetails=False):
        print('search.')
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
        print('selectNode.')
        while not self.isTerminal(node)[0]:
            if len(node.children)==0:
                return self.expand(node)
            elif random.uniform(0, 1) < 0.5:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                if node.isFullyExpanded():
                    node = self.getBestChild(node, self.explorationConstant)
                else:
                    return self.expand(node)
        return node

    def expand(self, node):
        print('expand.')
        while True:
            actions = self.getPossibleActions(node, self.treePolicy)
            assert actions is not None
            action = random.choice(actions)
            if len(action)==1:
                action = self.renderer.convertAction(action)
            if tuple(action) not in node.children:
                newNode = Node(self.renderer.numObjects, node.takeAction(action), node)
                node.children[tuple(action)] = newNode
                return newNode

    def getReward(self, table):
        print('getReward.')
        rgb = self.renderer.getRGB(table)
        s = torch.Tensor(rgb[None, :]).permute([0,3,1,2]).cuda()
        reward = self.VNet(s).cpu().detach().numpy()[0]
        return reward

    def backpropogate(self, node, reward):
        print('backpropagate.')
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def executeRound(self):
        print('executeRound.')
        node = self.selectNode(self.root)
        reward = self.rollout(node)
        self.backpropogate(node, reward)

    def getBestChild(self, node, explorationValue):
        print('getBestChild.')
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
        print('getPossibleActions.')
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
                s = torch.Tensor(np.array(states)).permute([0,3,1,2]).cuda()
                p = torch.Tensor(np.array(objectPatches)).permute([0,3,1,2]).cuda()
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
                                    np.arange(nb), np.arange(th), np.arange(tw)
                                    )).T.reshape(-1, 3)
                nextStates = []
                for action in allPossibleActions:
                    nextState = self.renderer.getRGB(node.takeAction(action))
                    nextStates.append(nextState)
                s = torch.Tensor(np.array(nextStates)).permute([0,3,1,2]).cuda()
                values = self.VNet(s).cpu().detach().numpy()
                possibleIdx = values > self.thresholdV
                node.setActions(allPossibleActions[possibleIdx])
            if needValues:
                return node,actionCandidates, values[possibleIdx]
            else:
                return node.actionCandidates
            
        elif policy=='random':
            if len(node.actionCandidates)==0:
                nb = self.renderer.numObjects
                th, tw = self.renderer.tableSize
                allPossibleActions = np.array(np.meshgrid(
                                    np.arange(nb), np.arange(th), np.arange(tw)
                                    )).T.reshape(-1, 3)
                node.setActions(allPossibleActions)
            return node.actionCandidates

    def isTerminal(self, node, table=None, checkReward=False):
        #print('isTerminal')
        if table is None:
            if node.depth >= self.maxDepth:
                return True, 0.0
            table = node.table
        for o in range(1, self.renderer.numObjects+1):
            if len(np.where(table==o)[0])==0:
                return True, 0.0
        if checkReward:
            reward = self.getReward(table)
            if reward > self.thresholdSuccess:
                return True, reward
            else:
                return False, reward
        return False, 0.0

    def noStepPolicy(self, node):
        print('noStepPolicy.')
        st = time.time()
        states = [self.renderer.getRGB(node.table)]
        s = torch.Tensor(np.array(states)).permute([0,3,1,2]).cuda()
        rewards = self.VNet(s).cpu().detach().numpy()
        maxReward = np.max(rewards)
        et = time.time()
        print(et - st, 'seconds.')
        return maxReward

    def oneStepPolicy(self, node):
        print('oneStepPolicy.')
        st = time.time()
        nb = self.renderer.numObjects
        th, tw = self.renderer.tableSize
        allPossibleActions = np.array(np.meshgrid(
                            np.arange(nb), np.arange(th), np.arange(tw)
                            )).T.reshape(-1, 3)
        states = [self.renderer.getRGB(node.table)]
        for action in allPossibleActions:
            newTable = node.takeAction(action)
            newNode = Node(self.renderer.numObjects, newTable)
            states.append(self.renderer.getRGB(newNode.table))

        s = torch.Tensor(np.array(states)).permute([0,3,1,2]).cuda()
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
        et = time.time()
        print(et - st, 'seconds.')
        return maxReward

    def randomPolicy(self, node):
        print('randomPolicy.')
        st = time.time()
        nb = self.renderer.numObjects
        th, tw = self.renderer.tableSize
        allPossibleActions = np.array(np.meshgrid(
                            np.arange(nb), np.arange(th), np.arange(tw)
                            )).T.reshape(-1, 3)
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
        s = torch.Tensor(np.array(states)).permute([0,3,1,2]).cuda()
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
        et = time.time()
        print(et - st, 'seconds.')
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
        s = torch.Tensor(np.array(states)).permute([0,3,1,2]).cuda()
        rewards = self.VNet(s).cpu().detach().numpy()
        maxReward = np.max(rewards)
        return maxReward

    def greedyPolicyWithV(self, node):
        state = self.renderer.getRGB(node.table)
        s = torch.Tensor(state).permute([0,3,1,2]).cuda()
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

if __name__=='__main__':
    dataDir = '/ssd/disk/ur5_tidying_data/pybullet_single_bg/images'
    rgbList = sorted([os.path.join(dataDir, r) for r in os.listdir(dataDir) if r.startswith('rgb')])
    segList = sorted([os.path.join(dataDir, s) for s in os.listdir(dataDir) if s.startswith('seg')])

    dataIndex = 0
    initRgb = np.array(Image.open(rgbList[dataIndex]))
    initSeg = np.flip(np.load(segList[dataIndex]), 0)
    renderer = Renderer(tableSize=(10, 10), imageSize=initRgb.shape[:2], cropSize=(48, 48))
    # plt.imshow(initSeg)
    # plt.show()
    searcher = MCTS(iterationLimit=2000, renderer=renderer, maxDepth=8,
                    rolloutPolicy='nostep', treePolicy='random')
    initTable = searcher.reset(initRgb, initSeg)
    print(initTable)
    table = initTable

    # torch simple convolution layer network
    vnet = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64*60*60, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
        nn.Sigmoid()
    ).cuda()
    searcher.setVNet(vnet)

    print("--------------------------------")    
    for s in range(10):
        resultDict = searcher.search(table=table, needDetails=True)
        print("Num Children: %d"%len(searcher.root.children))
        for i, c in enumerate(searcher.root.children):
            print(i, c, searcher.root.children[c])
        action = resultDict['action']
        nextTable = searcher.root.takeAction(action)
        print("Best Action:", action)
        print("Best Child: \n %s"%nextTable)
        print("--------------------------------")    
        tableRgb = renderer.getRGB(nextTable)
        # plt.imshow(tableRgb)
        # plt.show()
        table = copy.deepcopy(nextTable)
        terminal, reward = searcher.isTerminal(None, table, checkReward=True)
        if terminal:
            print("Arrived at the final state:")
            print("Score:", reward)
            print(table)
            print("--------------------------------")    
            print("--------------------------------")    
            break

