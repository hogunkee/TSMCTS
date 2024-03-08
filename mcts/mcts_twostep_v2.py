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

import torch
from data_loader import TabletopTemplateDataset
from utils import loadRewardFunction, Renderer, getGraph, visualizeGraph, summaryGraph

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp/pybullet_ur5_robotiq'))
from custom_env import TableTopTidyingUpEnv, get_contact_objects
from utilities import Camera, Camera_front_top

        
class NodePick(object):
    def __init__(self, numObjects, table, parent=None):
        self.type = 'pick'
        self.table = table
        self.parent = parent
        self.numObjcts = numObjects
        if parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

        self.numVisits = 0
        self.totalReward = 0.
        self.children = {}
        self.actionCandidates = np.arange(1, numObjects+1).tolist()
        self.actionProb = None
        self.terminal = False
    
    def takeAction(self, obj_pick):
        posMap, rotMap = self.table
        newPosMap = copy.deepcopy(posMap)
        newRotMap = copy.deepcopy(rotMap)
        newPosMap[posMap==obj_pick] = 0
        newRotMap[posMap==obj_pick] = 0
        newTable = [newPosMap, newRotMap]
        return newTable
    
    def setActions(self, actionCandidates, actionProb=None):
        self.actionCandidates = actionCandidates
        if actionProb is not None:
            self.actionProb = actionProb
            if False: # blurring
                newActionProb = np.zeros_like(actionProb)
                for i in range(len(actionProb)):
                    ap = actionProb[i]
                    kernel = np.ones((2, 2))
                    ap_blur = cv2.dilate(cv2.erode(ap, kernel), kernel)
                    newActionProb[i] = ap_blur
                self.actionProb = newActionProb

    def isFullyExpanded(self):
        return len(self.children)!=0 and len(self.children)==len(self.actionCandidates)

    def __str__(self):
        s=[]
        s.append("Reward: %s"%(self.totalReward))
        s.append("Visits: %d"%(self.numVisits))
        s.append("Terminal: %s"%(self.terminal))
        s.append("Children: %d"%(len(self.children.keys())))
        return "%s: %s"%(self.__class__.__name__, ' / '.join(s))


class NodePlace(object):
    def __init__(self, numObjects, table, selected, exceptPlace=None, parent=None):
        self.type = 'place'
        self.table = table
        self.selected = selected
        self.exceptPlace = exceptPlace
        self.parent = parent
        self.numObjcts = numObjects
        if parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

        self.numVisits = 0
        self.totalReward = 0.
        self.children = {}
        self.actionCandidates = []
        self.actionProb = None
        self.terminal = False
    
    def takeAction(self, placement):
        py, px, rot = placement
        posMap, rotMap = self.table
        newPosMap = copy.deepcopy(posMap)
        newRotMap = copy.deepcopy(rotMap)
        #newPosMap[posMap==self.selected] = 0
        #newRotMap[posMap==self.selected] = 0
        newPosMap[py, px] = self.selected
        newRotMap[py, px] = rot
        newTable = [newPosMap, newRotMap]
        return newTable
    
    def setActions(self, actionCandidates, actionProb=None):
        self.actionCandidates = actionCandidates
        if actionProb is not None:
            self.actionProb = actionProb
            if False: # blurring
                newActionProb = np.zeros_like(actionProb)
                for i in range(len(actionProb)):
                    ap = actionProb[i]
                    kernel = np.ones((2, 2))
                    ap_blur = cv2.dilate(cv2.erode(ap, kernel), kernel)
                    newActionProb[i] = ap_blur
                self.actionProb = newActionProb

    def isFullyExpanded(self):
        return len(self.children)!=0 and len(self.children)==len(self.actionCandidates)

    def __str__(self):
        s=[]
        s.append("Select: %d"%(self.selected))
        s.append("Reward: %s"%(self.totalReward))
        s.append("Visits: %d"%(self.numVisits))
        #s.append("Terminal: %s"%(self.terminal))
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
        self.binaryReward = args.binary_reward

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
        self.root = NodePick(self.renderer.numObjects, table)
        if table is not None:
            self.root = NodePick(self.renderer.numObjects, table)
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()
        bestPickChild = self.getBestChild(self.root, explorationValue=0.)
        pickAction=(action for action, node in self.root.children.items() if node is bestPickChild).__next__()
        bestPlaceChild = self.getBestChild(bestPickChild, explorationValue=0.)
        placeAction=(action for action, node in bestPickChild.children.items() if node is bestPlaceChild).__next__()
        action = [pickAction, *placeAction]
        if needDetails:
            return {
                    "action": action, 
                    "expectedPickReward": bestPickChild.totalReward / bestPickChild.numVisits,
                    "expectedPlaceReward": bestPlaceChild.totalReward / bestPlaceChild.numVisits
                    }
        else:
            return action

    def selectNode(self, nodePick):
        # print('selectNode.')
        while not nodePick.terminal: # self.isTerminal(node)[0]:
            assert nodePick.type=='pick'
            if len(nodePick.children)==0:
                return self.expand(nodePick)
            elif nodePick.isFullyExpanded() or random.uniform(0, 1) < 0.5:
                nodePlace = self.getBestChild(nodePick, self.explorationConstant)
                if len(nodePlace.children)==0:
                    return self.expandPlace(nodePlace)
                elif nodePlace.isFullyExpanded() or random.uniform(0, 1) < 0.5:
                    nodePick = self.getBestChild(nodePlace, self.explorationConstant)
                else:
                    return self.expandPlace(nodePlace)
            else:
                return self.expand(nodePick)
        return nodePick

    def expand(self, nodePick):
        # print('expand.')
        newNodePlace = self.expandPick(nodePick)
        newNodePick = self.expandPlace(newNodePlace)
        return newNodePick
    
    def expandPick(self, node):
        # print('expandPick.')
        actions, prob = self.getPossibleActions(node, self.treePolicy)
        # print('Num possible actions:', len(actions))
        
        action = np.random.choice([a for a in actions if a not in node.children])
        ey, ex = np.where(node.table[0]==action)
        if len(ey)==0 and len(ex)==0:
            newNode = NodePlace(self.renderer.numObjects, node.takeAction(action), action, None, node)
        else:
            ey, ex = np.mean(ey).astype(int), np.mean(ex).astype(int)
            newNode = NodePlace(self.renderer.numObjects, node.takeAction(action), action, (ey, ex), node)
        node.children[action] = newNode
        return newNode

    def expandPlace(self, node):
        # print('expandPlace.')
        actions, prob = self.getPossibleActions(node, self.treePolicy)
        actions = [tuple(a) for a in actions]
        # print('Num possible actions:', len(actions))
        
        action = random.choice([a for a in actions if a not in node.children])
        newNode = NodePick(self.renderer.numObjects, node.takeAction(action), node)
        node.children[action] = newNode
        return newNode

    def getReward(self, tables):
        # print('getReward.')
        states = []
        for table in tables:
            rgb = self.renderer.getRGB(table)
            states.append(rgb)
        s = torch.Tensor(np.array(states)/255.).permute([0,3,1,2]).cuda()
        if self.preProcess is not None:
            s = self.preProcess(s)

        if len(states) > self.batchSize:
            rewards = []
            numBatches = len(states)//self.batchSize
            if len(states)%self.batchSize > 0:
                numBatches += 1
            for b in range(numBatches):
                batchS = s[b*self.batchSize:(b+1)*self.batchSize]
                batchRewards = self.VNet(batchS).cpu().detach().numpy()
                rewards.append(batchRewards)
            rewards = np.concatenate(rewards)
        else:
            rewards = self.VNet(s).cpu().detach().numpy()
        return rewards.reshape(-1)

    def backpropogate(self, node, reward):
        # print('backpropagate.')
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def executeRound(self):
        # print('executeRound.')
        node = self.selectNode(self.root)
        assert node.type=='pick'
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
        if policy=='random':
            if len(node.actionCandidates)==0:
                if node.type=='pick':
                    nb = self.renderer.numObjects
                    actionCandidates = np.arange(1, nb+1)
                    probMap = np.ones(self.renderer.numObjects)
                    probMap /= np.sum(probMap)
                    node.setActions(actionCandidates, probMap)
                elif node.type=='place':
                    nb = self.renderer.numObjects
                    th, tw = self.renderer.tableSize
                    allPossibleActions = np.array(np.meshgrid(
                                    np.arange(th), np.arange(tw), np.arange(1,3)
                                    )).T.reshape(-1, 3)
                    if node.exceptPlace is None:
                        actionCandidates = [a for a in allPossibleActions 
                                            if node.table[0][a[0], a[1]]==0]
                    else:
                        actionCandidates = [a for a in allPossibleActions 
                                            if node.table[0][a[0], a[1]]==0 and 
                                            tuple(a[:2])!=tuple(node.exceptPlace)]
                    probMap = np.ones([th, tw])
                    probMap /= np.sum(probMap, axis=(0,1), keepdims=True)
                    node.setActions(actionCandidates, probMap)
            return node.actionCandidates, node.actionProb
        else:
            raise NotImplementedError

    def isTerminal(self, node, table=None, checkReward=False):
        # print('isTerminal')
        terminal = False
        reward = 0.0
        if table is None:
            table = node.table
        # check collision and reward for Pick Nodes
        assert node is None or node.type=='pick'
        collision = self.renderer.checkCollision(table)
        if collision:
            reward = -1.0
            terminal = True
        elif checkReward:
            reward = self.getReward([table])[0]
            if reward > self.thresholdSuccess:
                terminal = True
        # check depth
        if node is not None:
            if node.depth >= self.maxDepth:
                terminal = True
            node.terminal = terminal
        return terminal, reward

    def noStepPolicy(self, node):
        # print('noStepPolicy.')
        # st = time.time()
        terminal, reward = self.isTerminal(node, checkReward=True)
        if self.binaryReward:
            if reward > self.thresholdSuccess:
                reward = 1.0
            elif reward == -1.0:
                reward = -1.0
            else:
                reward = 0.0
        maxReward = reward
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
        tables = [np.copy(node.table)]
        while not self.isTerminal(node)[0]:
            try:
                action = random.choice(allPossibleActions)
            except IndexError:
                raise Exception("Non-terminal state has no possible actions: " + str(state))
            newTable = node.takeAction(action)
            if node.type=='pick':
                ey, ex = np.where(node.table[0]==action)
                if len(ey)==0 and len(ex)==0:
                    newNode = NodePlace(self.renderer.numObjects, newTable, action, None, node)
                else:
                    ey, ex = np.mean(ey).astype(int), np.mean(ex).astype(int)
                    newNode = NodePlace(self.renderer.numObjects, newTable, action, (ey, ex), node)
            else:
                newNode = NodePick(self.renderer.numObjects, newTable, node)
            node = newNode
            if node.type=='pick':   # skip the place-node
                tables.append(np.copy(node.table))
        rewards = self.getReward(tables)
        maxReward = np.max(rewards)
        # et = time.time()
        # print(et - st, 'seconds.')
        return maxReward


def setupEnvironment(objects, args):
    camera_top = Camera((0, 0, 1.45), 0.02, 2, (480, 360), 60)
    camera_front_top = Camera_front_top((0.5, 0, 1.3), 0.02, 2, (480, 360), 60)
    
    data_dir = args.data_dir
    objects_cfg = { 'paths': {
            'pybullet_object_path' : os.path.join(data_dir, 'pybullet-URDF-models/urdf_models/models'),
            'ycb_object_path' : os.path.join(data_dir, 'YCB_dataset'),
            'housecat_object_path' : os.path.join(data_dir, 'housecat6d/obj_models_small_size_final'),
        },
        'split' : 'inference' #'train'
    }
    
    gui_on = not args.gui_off
    env = TableTopTidyingUpEnv(objects_cfg, camera_top, camera_front_top, vis=gui_on, num_objs=args.num_objects, gripper_type='85')
    p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off
    p.addUserDebugLine([0, -0.5, 0], [0, -0.5, 1.1], [0, 1, 0])

    env.reset()
    env.spawn_objects(objects)
    
    env.arrange_objects(random=True)
    return env


if __name__=='__main__':
    parser = ArgumentParser()
    # Data directory
    parser.add_argument('--data-dir', type=str, default='/ssd/disk')
    # Inference
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument('--num-objects', type=int, default=4)
    parser.add_argument('--num-scenes', type=int, default=10)
    parser.add_argument('--H', type=int, default=12)
    parser.add_argument('--W', type=int, default=15)
    parser.add_argument('--crop-size', type=int, default=128) #96
    parser.add_argument('--gui-off', action="store_true")
    parser.add_argument('--visualize-graph', action="store_true")
    # MCTS
    parser.add_argument('--time-limit', type=int, default=None)
    parser.add_argument('--iteration-limit', type=int, default=10000)
    parser.add_argument('--max-depth', type=int, default=14)
    parser.add_argument('--rollout-policy', type=str, default='nostep')
    parser.add_argument('--tree-policy', type=str, default='random')
    parser.add_argument('--threshold-success', type=float, default=0.9) #0.85
    parser.add_argument('--threshold-q', type=float, default=0.5)
    parser.add_argument('--threshold-v', type=float, default=0.5)
    parser.add_argument('--threshold-prob', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--binary-reward', action="store_true")
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

    # Random seed
    seed = args.seed
    if seed is not None:
        print("Random seed: %d"%seed)
        logger.info("Random seed: %d"%seed)
        random.seed(seed)
        np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # Environment setup
    objects = ['bowl', 'can_drink', 'plate', 'marker', 'soap_dish', 'book', 'remote', 'fork', 'knife', 'spoon', 'teapot', 'cup']
    objects = [(o, 'medium') for o in objects]
    #objects = [('bowl', 'medium'), ('can_drink','medium'), ('plate','medium'), ('marker', 'medium'), ('soap_dish', 'medium'), ('book', 'medium'), ('remote', 'medium')]
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
        searcher.setPolicyNet(pnet)
    
    success = 0
    for sidx in range(args.num_scenes):
        if seed is not None: np.random.seed(seed + sidx)
        # setup logger
        os.makedirs('data/twostep-%s/scene-%d'%(log_name, sidx), exist_ok=True)
        with open('data/twostep-%s/config.json'%log_name, 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(name)s -\n%(message)s')
        file_handler = logging.FileHandler('data/twostep-%s/scene-%d/mcts.log'%(log_name, sidx))
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
        plt.savefig('data/twostep-%s/scene-%d/initial.png'%(log_name, sidx))
        initTable = searcher.reset(initRgb, initSeg)
        print('initTable: \n %s' % initTable[0])
        logger.info('initTable: \n %s' % initTable[0])
        table = initTable

        print("--------------------------------")
        logger.info('-'*50)
        for step in range(10):
            resultDict = searcher.search(table=table, needDetails=True)
            print("Num Children: %d"%len(searcher.root.children))
            logger.info("Num Children: %d"%len(searcher.root.children))
            action = resultDict['action']

            summary = summaryGraph(searcher.root)
            if args.visualize_graph:
                graph = getGraph(searcher.root)
                fig = visualizeGraph(graph, title='MCTS two-step')
                fig.show()
            print(summary)
            logger.info(summary)
            
            # action probability
            actionProb = searcher.root.actionProb
            if actionProb is not None:
                actionProb[actionProb>args.threshold_prob] += 0.5
                plt.imshow(np.mean(actionProb, axis=0))
                plt.savefig('data/twostep-%s/scene-%d/actionprob_%d.png'%(log_name, sidx, step))

            # expected result in mcts #
            pick = action[0]
            place = action[1:]
            bestPickChild = (node for action, node in searcher.root.children.items() if action==pick).__next__()
            bestPlaceChild = (node for action, node in bestPickChild.children.items() if action==tuple(place)).__next__()
            print("Children of the root node:")
            logger.info("Children of the root node:")
            for c in sorted(list(searcher.root.children.keys())):
                print(searcher.root.children[c])
                logger.info(str(searcher.root.children[c]))
            print("Children of the best child pick node:")
            logger.info("Children of the best child pick node:")
            for i, c in enumerate(sorted(list(bestPickChild.children.keys()))):
                print(i, c, bestPickChild.children[c])
                logger.info(f"{i} {c} {str(bestPickChild.children[c])}")
            nextTable = bestPlaceChild.table
            # nextTable = searcher.root.takeAction(action)
            print("Best Action:", action)
            print("Best Child: \n %s"%nextTable[0])
            logger.info("Best Action: %s"%str(action))
            logger.info("Best Child: \n %s"%nextTable[0])
            if True:
                nextCollision = renderer.checkCollision(nextTable)
                logger.info("Collision: %s"%nextCollision)
                logger.info("Save fig: scene-%d/expect_%d.png"%(sidx, step))
            tableRgb = renderer.getRGB(nextTable)
            plt.imshow(tableRgb)
            plt.savefig('data/twostep-%s/scene-%d/expect_%d.png'%(log_name, sidx, step))
            #plt.show()

            # simulation step in pybullet #
            target_object, target_position, rot_angle = renderer.convert_action(action)
            obs = env.step(target_object, target_position, rot_angle)
            currentRgb = obs[args.view]['rgb']
            currentSeg = obs[args.view]['segmentation']
            plt.imshow(currentRgb)
            plt.savefig('data/twostep-%s/scene-%d/real_%d.png'%(log_name, sidx, step))

            table = searcher.reset(currentRgb, currentSeg)
            if table is None:
                print("Scenario ended.")
                logger.info("Scenario ended.")
                break
            #table = copy.deepcopy(nextTable)
            print("Current state: \n %s"%table[0])
            logger.info("Current state: \n %s"%table[0])

            terminal, reward = searcher.isTerminal(None, table, checkReward=True)
            print("Current Score:", reward)
            print("--------------------------------")
            logger.info("Current Score: %f" %reward)
            logger.info("-"*50)
            if terminal:
                print("Arrived at the final state:")
                print("Score:", reward)
                if reward > args.threshold_success:
                    success += 1
                print(table[0])
                print("--------------------------------")
                print("--------------------------------")
                logger.info("Arrived at the final state:")
                logger.info("Score: %f"%reward)
                logger.info(table[0])
                plt.imshow(currentRgb)
                plt.savefig('data/twostep-%s/scene-%d/final.png'%(log_name, sidx))
                # plt.show()
                break
    print("Success rate: %.2f (%d/%d)"%(success/args.num_scenes, success, args.num_scenes))

