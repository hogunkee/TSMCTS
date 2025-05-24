#!/bin/bash
GPU_ID=$1
GPU_ID2=$2
TAG=mcts-uniform

OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=${GPU_ID} python mcts.py --iteration-limit 3000 --gui-off --data-dir /disk1/hogun --num-scenes 50 --seed 12345 --logging --rollout-policy nostep --tree-policy random --exploration 0.5 --use-template --object-split unseen --num-objects 0 --scenes B2,B5 --block-preaction --tag ${TAG} &
OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=${GPU_ID} python mcts.py --iteration-limit 3000 --gui-off --data-dir /disk1/hogun --num-scenes 50 --seed 12345 --logging --rollout-policy nostep --tree-policy random --exploration 0.5 --use-template --object-split unseen --num-objects 0 --scenes C4,C6,C12 --block-preaction --tag ${TAG} &
OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=${GPU_ID} python mcts.py --iteration-limit 3000 --gui-off --data-dir /disk1/hogun --num-scenes 50 --seed 12345 --logging --rollout-policy nostep --tree-policy random --exploration 0.5 --use-template --object-split unseen --num-objects 0 --scenes D5,D8,D11 --block-preaction --tag ${TAG} &
OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=${GPU_ID} python mcts.py --iteration-limit 3000 --gui-off --data-dir /disk1/hogun --num-scenes 50 --seed 12345 --logging --rollout-policy nostep --tree-policy random --exploration 0.5 --use-template --object-split unseen --num-objects 0 --scenes O3,O7 --block-preaction --tag ${TAG} &
OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=${GPU_ID2} python mcts.py --iteration-limit 3000 --gui-off --data-dir /disk1/hogun --num-scenes 50 --seed 12345 --logging --rollout-policy nostep --tree-policy random --exploration 0.5 --object-split unseen --num-objects 5 --block-preaction --tag ${TAG}
