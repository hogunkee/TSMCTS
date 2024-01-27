#!/usr/bin/env python
import random
import math
import hashlib
import logging
import argparse
import numpy as np


"""
A quick Monte Carlo Tree Search implementation.  For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf

The State is a game where you have NUM_TURNS and at turn i you can make
a choice from an integeter [-2,2,3,-3]*(NUM_TURNS+1-i).  So for example in a game of 4 turns, on turn for turn 1 you can can choose from [-8,8,12,-12], and on turn 2 you can choose from [-6,6,9,-9].  At each turn the choosen number is accumulated into a aggregation value.  The goal of the game is for the accumulated value to be as close to 0 as possible.

The game is not very interesting but it allows one to study MCTS which is.  Some features 
of the example by design are that moves do not commute and early mistakes are more costly.  

In particular there are two models of best child that one can use 

move = (obj, x, y)
obj = move // (resolution * resolution)
y = (move % (resolution * resolution)) // resolution
x = (move % (resolution * resolution)) % resolution
move index can be one of 0 ~ NB*RES*RES
"""

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration. 
SCALAR=1/(2*math.sqrt(2.0))
NUM_OBJ = 4
RESOLUTION = 5 #10

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')


class State():
    num_objects = NUM_OBJ
    resolution = RESOLUTION
    def __init__(self, table=None, moves=[]):
        if table is None:
            table = np.zeros([self.resolution, self.resolution], dtype=int)
        self.table = table
        self.moves = moves
        self.num_moves = self.num_objects * self.resolution * self.resolution

    def next_state(self):
        move_idx = np.random.choice(self.num_objects * self.resolution * self.resolution)
        obj = move_idx // (self.resolution * self.resolution) + 1
        py = (move_idx % (self.resolution * self.resolution)) // self.resolution
        px = (move_idx % (self.resolution * self.resolution)) % self.resolution
        nextmove = (obj, py, px)
        new_table = np.copy(self.table)
        new_table[new_table==obj] = 0
        new_table[py, px] = obj
        next = State(new_table, self.moves+[nextmove])
        return next

    def terminal(self):
        for o in range(1, self.num_objects+1):
            if len(np.where(self.table==o)[0])==0:
                return True
        alignment = ' '.join([str(o) for o in range(1, self.num_objects+1)])
        if alignment in str(self.table) or alignment[::-1] in str(self.table):
            return True
        if alignment in str(self.table.T) or alignment[::-1] in str(self.table.T):
            return True
        return False

    def reward(self):
        r = 0.0
        alignment = ' '.join([str(o) for o in range(1, self.num_objects+1)])
        if alignment in str(self.table) or alignment[::-1] in str(self.table):
            r = 1.0
        if alignment in str(self.table.T) or alignment[::-1] in str(self.table.T):
            r = 1.0
        return r

    def __hash__(self):
        return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(),16)

    def __eq__(self,other):
        if hash(self)==hash(other):
            return True
        return False

    def __repr__(self):
        s="Table:\n%s\nMoves: %s" %(str(self.table), self.moves)
        return s
    

class Node():
    def __init__(self, state, parent=None):
        self.visits=1
        self.reward=0.0    
        self.state=state
        self.children=[]
        self.parent=parent    

    def add_child(self,child_state):
        child=Node(child_state,self)
        self.children.append(child)

    def update(self,reward):
        self.reward+=reward
        self.visits+=1

    def fully_expanded(self, num_moves_lambda):
        num_moves = self.state.num_moves
        if num_moves_lambda != None:
          num_moves = num_moves_lambda(self)
        if len(self.children)==num_moves:
            return True
        return False

    def __repr__(self):
        s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
        return s

def UCTSEARCH(budget, root, num_moves_lambda=None):
    for iter in range(int(budget)):
        if iter%1000==999:
            #print("simulation: %d" %iter)
            #print(root)
            logger.info("simulation: %d"%iter)
            logger.info(root)
        front = TREEPOLICY(root, num_moves_lambda)
        reward = DEFAULTPOLICY(front.state)
        BACKUP(front,reward)
    return BESTCHILD(root,0)

def TREEPOLICY(node, num_moves_lambda):
    #a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
    while node.state.terminal()==False:
        if len(node.children)==0:
            return EXPAND(node)
        elif random.uniform(0,1)<.5:
            node=BESTCHILD(node,SCALAR)
        else:
            if node.fully_expanded(num_moves_lambda)==False:    
                return EXPAND(node)
            else:
                node=BESTCHILD(node,SCALAR)
    return node

def EXPAND(node):
    tried_children=[c.state for c in node.children]
    new_state=node.state.next_state()
    while new_state in tried_children and new_state.terminal()==False:
        new_state=node.state.next_state()
    node.add_child(new_state)
    return node.children[-1]

#current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node,scalar):
    bestscore=0.0
    bestchildren=[]
    for c in node.children:
        exploit=c.reward/c.visits
        explore=math.sqrt(2.0*math.log(node.visits)/float(c.visits))    
        score=exploit+scalar*explore
        if score==bestscore:
            bestchildren.append(c)
        if score>bestscore:
            bestchildren=[c]
            bestscore=score
    if len(bestchildren)==0:
        print("OOPS: no best child found, probably fatal")
        #logger.warning("OOPS: no best child found, probably fatal")
    return random.choice(bestchildren)

def DEFAULTPOLICY(state):
    while state.terminal()==False:
        state=state.next_state()
    return state.reward()

def BACKUP(node,reward):
    while node!=None:
        node.visits+=1
        node.reward+=reward
        node=node.parent
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='MCTS research code')
    parser.add_argument('--num_sims', action="store", required=True, type=int)
    parser.add_argument('--levels', action="store", required=True, type=int)
    args=parser.parse_args()
    
    random_table = np.zeros([RESOLUTION, RESOLUTION], dtype=int)
    poses = np.random.choice(RESOLUTION, [NUM_OBJ, 2])
    for o, p in enumerate(poses):
        random_table[p[0], p[1]] = o+1

    def num_moves_lambda(node):
        return 10

    print(random_table)
    current_node=Node(State(random_table))
    for l in range(args.levels):
        current_node=UCTSEARCH(args.num_sims/(l+1),current_node, num_moves_lambda)
        print("level %d"%l)
        print("Num Children: %d"%len(current_node.children))
        for i,c in enumerate(current_node.children):
            print(i,c)
        print("Best Child: %s"%current_node.state)
        
        print("--------------------------------")    
            
    
