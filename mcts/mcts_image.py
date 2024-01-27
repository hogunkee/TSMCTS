from __future__ import division

import time
import os
import copy
import math
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

NUM_OBJ = 4
RESOLUTION = 10

def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()

def convert_action(move_idx):
        obj = move_idx // (RESOLUTION * RESOLUTION) + 1
        py = (move_idx % (RESOLUTION * RESOLUTION)) // RESOLUTION
        px = (move_idx % (RESOLUTION * RESOLUTION)) % RESOLUTION
        return (obj, py, px)

class ObjectInfo():
    def __init__(self, rgb, segmap):
        self.num_obj = np.max(segmap).astype(int)
        self.res_ratio, self.offset = self.setup(rgb) 
        self.rgb = Image.fromarray(rgb)
        self.masks, self.centers = self.get_masks(segmap)

    def setup(self, rgb):
        img_res = rgb.shape[0]
        res_ratio = img_res // RESOLUTION
        offset = (img_res - res_ratio * RESOLUTION)//2
        offset += res_ratio//2
        return res_ratio, offset

    def get_masks(self, segmap):
        masks, centers = [], []
        for o in range(1, self.num_obj+1):
            # get the segmentation mask of each object #
            mask = Image.fromarray(255*(segmap==o).astype(np.uint8))
            masks.append(mask)
            # get the center of each object #
            py, px = np.where(segmap==o)
            cy, cx = np.mean(py), np.mean(px)
            center = (cy, cx)
            centers.append(center)
        return masks, centers

    def grid2rgb(self, table):
        new_rgb = Image.fromarray(np.zeros_like(np.array(self.rgb)))
        #new_rgb = Image.fromarray(np.zeros([RESOLUTION, RESOLUTION, 4], dtype=np.uint8))
        for o in range(1, self.num_obj+1):
            py, px = np.where(table==o)
            py, px = py[0], px[0]
            cy, cx = self.centers[o-1]
            ty, tx = np.array([py, px]) * self.res_ratio + self.offset - np.array([cy, cx])
            ty, tx = np.array([ty, tx]).astype(int)
            new_rgb.paste(self.rgb, (tx, ty), self.masks[o-1])
        return np.array(new_rgb)
    
    def segmap2grid(self, segmap):
        new_table = np.zeros([RESOLUTION, RESOLUTION])
        for o in range(1, self.num_obj+1):
            center = self.centers[o-1]
            gy, gx = ((np.array(center) - self.offset) // self.res_ratio).astype(int)
            new_table[gy, gx] = o
        return new_table

class State():
    num_objects = NUM_OBJ
    def __init__(self, table=None):
        if table is None:
            table = np.zeros([RESOLUTION, RESOLUTION], dtype=int)
        self.table = table
        self.num_moves = self.num_objects * RESOLUTION * RESOLUTION

    def takeAction(self, move_idx):
        obj, py, px = convert_action(move_idx)
        new_table = copy.deepcopy(self.table)
        new_table[new_table==obj] = 0
        new_table[py, px] = obj
        nextstate = State(new_table)
        return nextstate
    
    def getPossibleActions(self):
        return np.arange(self.num_moves)

    def isTerminal(self):
        for o in range(1, self.num_objects+1):
            if len(np.where(self.table==o)[0])==0:
                return True
        alignment = ' '.join([str(o) for o in range(1, self.num_objects+1)])
        if alignment in str(self.table) or alignment[::-1] in str(self.table):
            return True
        if alignment in str(self.table.T) or alignment[::-1] in str(self.table.T):
            return True
        return False

    def getReward(self):
        r = 0.0
        alignment = ' '.join([str(o) for o in range(1, self.num_objects+1)])
        if alignment in str(self.table) or alignment[::-1] in str(self.table):
            r = 1.0
        if alignment in str(self.table.T) or alignment[::-1] in str(self.table.T):
            r = 1.0
        return r

    def __repr__(self):
        s = str(self.table)
        return s

class TreeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __str__(self):
        s=[]
        s.append("totalReward: %s"%(self.totalReward))
        s.append("numVisits: %d"%(self.numVisits))
        s.append("isTerminal: %s"%(self.isTerminal))
        s.append("children: %d"%(len(self.children.keys())))
        return "%s: %s"%(self.__class__.__name__, ' / '.join(s))

class MCTS():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1/math.sqrt(2),
                 rolloutPolicy=randomPolicy):
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
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState, needDetails=False):
        self.root = TreeNode(initialState, None)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        action=(action for action, node in self.root.children.items() if node is bestChild).__next__()
        if needDetails:
            return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        else:
            return action

    def executeRound(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        node = self.selectNode(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if len(node.children)==0:
                return self.expand(node)
            elif random.uniform(0, 1) < 0.5:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                if node.isFullyExpanded:
                    node = self.getBestChild(node, self.explorationConstant)
                else:
                    return self.expand(node)
        return node
        #while not node.isTerminal:
        #    if node.isFullyExpanded:
        #        node = self.getBestChild(node, self.explorationConstant)
        #    else:
        #        return self.expand(node)
        #return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        while True:
            action = random.choice(actions)
            if action not in node.children:
                newNode = TreeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(node.children) == 50:
                #if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.totalReward / child.numVisits + explorationValue * \
                    math.sqrt(2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

if __name__=='__main__':
    random_table = np.zeros([RESOLUTION, RESOLUTION], dtype=int)
    poses = np.random.choice(RESOLUTION, [NUM_OBJ, 2])
    for o, p in enumerate(poses):
        random_table[p[0], p[1]] = o+1

    data_dir = '../samples/'
    rgb_list = sorted([os.path.join(data_dir, r) for r in os.listdir(data_dir) if r.startswith('rgb')])
    seg_list = sorted([os.path.join(data_dir, s) for s in os.listdir(data_dir) if s.startswith('seg')])
    init_rgb = np.array(Image.open(rgb_list[0]))
    init_seg = np.flip(np.load(seg_list[0]), 0)
    obj_info = ObjectInfo(init_rgb, init_seg)

    plt.imshow(init_seg)
    plt.show()
    init_table = obj_info.segmap2grid(init_seg)

    state = State(init_table)
    #searcher = MCTS(iterationLimit=10000)
    searcher = MCTS(timeLimit=2000)
    print("--------------------------------")    
    print("Initial State:")
    print(state)
    for s in range(10):
        #bestAction = searcher.search(initialState=state)
        resultDict = searcher.search(initialState=state, needDetails=True)
        print("Num Children: %d"%len(searcher.root.children))
        for i, c in enumerate(searcher.root.children):
            print(i, convert_action(c), searcher.root.children[c])
        action = resultDict['action']
        print("Best Action:", convert_action(action))
        next_state = state.takeAction(resultDict['action'])
        print("Best Child: \n %s"%next_state)
        print("--------------------------------")    
        table_rgb = obj_info.grid2rgb(next_state.table)
        plt.imshow(table_rgb)
        plt.show()
        state = copy.deepcopy(next_state)
        if state.isTerminal():
            print("Arrived at the final state:")
            print(state)
            print("--------------------------------")    
            print("--------------------------------")    
            break

