#!/bin/bash
GPU_ID=$1
GPU_ID2=$2
SEED=$3
NUMEP=$4
TAG=greedy

OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=${GPU_ID} python greedy.py --iteration-limit 3000 --gui-off --data-dir /disk1/hogun --num-scenes ${NUMEP} --seed ${SEED} --logging --rollout-policy nostep --tree-policy random --blurring 1 --exploration 0.5 --use-template --object-split unseen --num-objects 0 --scenes B2,B5 --policy-version 1 --prob-expand 0 --threshold-success 0.9 --block-preaction --tag ${TAG} &
OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=${GPU_ID} python greedy.py --iteration-limit 3000 --gui-off --data-dir /disk1/hogun --num-scenes ${NUMEP} --seed ${SEED} --logging --rollout-policy nostep --tree-policy random --blurring 1 --exploration 0.5 --use-template --object-split unseen --num-objects 0 --scenes C4,C6,C12 --policy-version 1 --prob-expand 0 --threshold-success 0.9 --block-preaction --tag ${TAG} &
OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=${GPU_ID} python greedy.py --iteration-limit 3000 --gui-off --data-dir /disk1/hogun --num-scenes ${NUMEP} --seed ${SEED} --logging --rollout-policy nostep --tree-policy random --blurring 1 --exploration 0.5 --use-template --object-split unseen --num-objects 0 --scenes D5,D8,D11 --policy-version 1 --prob-expand 0 --threshold-success 0.9 --block-preaction --tag ${TAG} &
OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=${GPU_ID} python greedy.py --iteration-limit 3000 --gui-off --data-dir /disk1/hogun --num-scenes ${NUMEP} --seed ${SEED} --logging --rollout-policy nostep --tree-policy random --blurring 1 --exploration 0.5 --use-template --object-split unseen --num-objects 0 --scenes O3,O7 --policy-version 1 --prob-expand 0 --threshold-success 0.9 --block-preaction --tag ${TAG} &
OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=${GPU_ID2} python greedy.py --iteration-limit 3000 --gui-off --data-dir /disk1/hogun --num-scenes ${NUMEP} --seed ${SEED} --logging --rollout-policy nostep --tree-policy random --blurring 1 --exploration 0.5 --object-split unseen --num-objects 5 --policy-version 1 --prob-expand 0 --threshold-success 0.9 --block-preaction --tag ${TAG}
