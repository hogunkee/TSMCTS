import copy
import time
import torch
import numpy as np
import logging
import os
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from structformer.data.tokenizer import Tokenizer
from structformer.evaluation.test_object_selection_network import ObjectSelectionInference
from structformer.evaluation.test_structformer import PriorInference
from structformer.utils.rearrangement import show_pcs_with_predictions, get_initial_scene_idxs, evaluate_target_object_predictions, save_img, show_pcs_with_labels, test_new_vis, show_pcs
from structformer.evaluation.inference import PointCloudRearrangement

import json
import datetime
import random
import cv2
import pybullet as p
from matplotlib import pyplot as plt
from pc_utils import get_raw_data #, setupEnvironment
from transform_utils import mat2quat


# tabletop environment
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp/pybullet_ur5_robotiq'))
from custom_env import get_contact_objects, quaternion_multiply
sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp'))
from collect_template_list import scene_list
sys.path.append(os.path.join(FILE_PATH, '..', 'mcts'))
from mcts import setupEnvironment
from utils import suppress_stdout, loadRewardFunction


def getReward(rgb, rewardNet, preProcess):
    # print('getReward.')
    states = [rgb] 
    s_reward = torch.Tensor(np.array(states)/255.).permute([0,3,1,2]).cuda()
    s_reward = preProcess(s_reward)

    rewards = rewardNet(s_reward).cpu().detach().numpy()
    return rewards.reshape(-1)

def run_demo(object_selection_model_dir, pose_generation_model_dir, dirs_config, args):
    """
    Run a simple demo. Creates the object selection inference model, pose generation model,
    and so on. Requires paths to the model + config directories.
    """
    # Logger
    now = datetime.datetime.now()
    log_name = now.strftime("%m%d_%H%M")
    log_name += '-' + args.view
    # logname = 'SF-' + log_name
    if args.logging:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    # load reward networks
    if args.get_reward:
        model_path = args.reward_model_path
        rewardNet, preprocess = loadRewardFunction(model_path)

    # Random seed
    seed = args.seed
    if seed is not None:
        print("Random seed: %d"%seed)
        random.seed(seed)
        np.random.seed(seed)

    # Environment setup
    with suppress_stdout():
        env = setupEnvironment(args, log_name)
    scenes = sorted(list(scene_list.keys()))
    if args.use_template:
        scenes = sorted(list(scene_list.keys()))
        if args.scenes=='':
            if args.scene_split=='unseen':
                scenes = [s for s in scenes if s in ['B2', 'B5', 'C4', 'C6', 'C12', 'D5', 'D8', 'D11', 'O3', 'O7']]
            elif args.scene_split=='seen':
                scenes = [s for s in scenes if s not in ['B2', 'B5', 'C4', 'C6', 'C12', 'D5', 'D8', 'D11', 'O3', 'O7']]
        else:
            scenes = args.scenes.split(',')
        if args.inorder:
            selected_scene = scenes[0]
            metrics = {}
            for s in scenes:
                metrics[s] = {}
        else:
            selected_scene = random.choice(scenes)
        print('Selected scene: %s' %selected_scene)

        objects = scene_list[selected_scene]
        #sizes = [random.choice(['small', 'medium', 'large']) for o in objects]
        sizes = []
        for i in range(len(objects)):
            if 'small' in objects[i]:
                sizes.append('small')
                objects[i] = objects[i].replace('small_', '')
            elif 'large' in objects[i]:
                sizes.append('large')
                objects[i] = objects[i].replace('large_', '')
            else:
                sizes.append('medium')
        objects = [[objects[i], sizes[i]] for i in range(len(objects))]
        
    else:
        objects = ['book', 'bowl', 'can_drink', 'can_food', 'cleanser', 'cup', 'fork', 'fruit', 
                'glass', 'glue', 'knife', 'lotion', 'marker', 'plate', 'remote', 'scissors', 
                'shampoo', 'soap_dish', 'spoon', 'stapler', 'timer', 'toothpaste']
        objects = [(o, 'medium') for o in objects]
    if args.num_objects==0: # use the template
        selected_objects = objects
    else: # random sampling
        if len(objects) < args.num_objects:
            selected_objects = [objects[i] for i in np.random.choice(len(objects), args.num_objects, replace=True)]
        else:
            selected_objects = [objects[i] for i in np.random.choice(len(objects), args.num_objects, replace=False)]
    env.spawn_objects(selected_objects)
    env.arrange_objects(random=True)

    object_selection_inference = ObjectSelectionInference(object_selection_model_dir, dirs_cfg)
    pose_generation_inference = PriorInference(pose_generation_model_dir, dirs_cfg)

    cmap = plt.cm.get_cmap('hsv', 20)
    cmap = np.array([cmap(i) for i in range(20)])
    cmap = (255*cmap).astype(np.uint8)

    success = 0
    success_eplen = []
    log_dir = 'data/SFO'
    bar = range(args.num_scenes)
    for sidx in bar:
        if args.logging: 
            # bar.set_description("Episode %d/%d"%(sidx, args.num_scenes))
            os.makedirs('%s-%s/scene-%d'%(log_dir, log_name, sidx), exist_ok=True)
            with open('%s-%s/config.json'%(log_dir, log_name), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

            logger.handlers.clear()
            formatter = logging.Formatter('%(asctime)s - %(name)s -\n%(message)s')
            file_handler = logging.FileHandler('%s-%s/scene-%d/structformer.log'%(log_dir, log_name, sidx))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if seed is not None: 
            random.seed(seed + sidx)
            np.random.seed(seed + sidx)
        
        # Initial state
        with suppress_stdout():
            obs = env.reset()
        if args.use_template:
            if args.inorder:
                selected_scene = scenes[sidx%len(scenes)]
            else:
                selected_scene = random.choice(scenes)
            print('Selected scene: %s' %selected_scene)

            objects = scene_list[selected_scene]
            #sizes = [random.choice(['small', 'medium', 'large']) for o in objects]
            sizes = []
            for i in range(len(objects)):
                if 'small' in objects[i]:
                    sizes.append('small')
                    objects[i] = objects[i].replace('small_', '')
                elif 'large' in objects[i]:
                    sizes.append('large')
                    objects[i] = objects[i].replace('large_', '')
                else:
                    sizes.append('medium')
            objects = [[objects[i], sizes[i]] for i in range(len(objects))]
            
        if args.num_objects==0:
            selected_objects = objects
        else:
            if len(objects) < args.num_objects:
                selected_objects = [objects[i] for i in np.random.choice(len(objects), args.num_objects, replace=True)]
            else:
                selected_objects = [objects[i] for i in np.random.choice(len(objects), args.num_objects, replace=False)]
        env.spawn_objects(selected_objects)
        while True:
            is_occluded = False
            is_collision = False
            env.arrange_objects(random=True)
            obs = env.get_observation()
            initRgb = obs['top']['rgb']
            initSeg = obs['top']['segmentation']
            # Check occlusions
            for o in range(len(selected_objects)):
                # get the segmentation mask of each object #
                mask = (initSeg==o+4).astype(float)
                if mask.sum()==0:
                    print("Object %d is occluded."%o)
                    is_occluded = True
                    break
            # Check collision
            contact_objects = get_contact_objects()
            contact_objects = [c for c in list(get_contact_objects()) if 1 not in c and 2 not in c]
            if len(contact_objects) > 0:
                print("Collision detected.")
                print(contact_objects)
                is_collision = True
            if is_occluded or is_collision:
                continue
            else:
                break
        print('Objects: %s' %[o for o,s in selected_objects])

        initRgbFront = obs['front']['rgb']
        initRgbNV = obs['nv-'+args.view]['rgb']
        initSegNV = obs['nv-'+args.view]['segmentation']
        initRgbFrontNV = obs['nv-front']['rgb']

        if args.logging:
            cv2.imwrite('%s-%s/scene-%d/top_initial.png'%(log_dir, log_name, sidx), cv2.cvtColor(initRgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite('%s-%s/scene-%d/front_initial.png'%(log_dir, log_name, sidx), cv2.cvtColor(initRgbFront, cv2.COLOR_RGB2BGR))
            cv2.imwrite('%s-%s/scene-%d/nv_top_initial.png'%(log_dir, log_name, sidx), cv2.cvtColor(initRgbNV, cv2.COLOR_RGB2BGR))
            cv2.imwrite('%s-%s/scene-%d/nv_front_initial.png'%(log_dir, log_name, sidx), cv2.cvtColor(initRgbFrontNV, cv2.COLOR_RGB2BGR))
            cv2.imwrite('%s-%s/scene-%d/top_seg_init.png'%(log_dir, log_name, sidx), cv2.cvtColor(cmap[initSeg.astype(int)], cv2.COLOR_RGB2BGR))
            cv2.imwrite('%s-%s/scene-%d/top_seg_init_nv.png'%(log_dir, log_name, sidx), cv2.cvtColor(cmap[initSegNV.astype(int)], cv2.COLOR_RGB2BGR))
        
        structure_param = {'length': np.random.uniform(0.25, 0.4),
                           'position': np.random.uniform(0.35, 0.65)}
        print("--------------------------------")

        init_datum = get_raw_data(env.get_observation(), env, structure_param, view=args.view)

        object_selection_structured_sentence = [('dinner', 'scene'), ('PAD',), ('PAD',), ('PAD',)]
        structure_specification_structured_sentence = [('dinner', 'shape'),
                                                    (0.0, 'rotation'),
                                                    (structure_param['position'], 'position_x'),
                                                    (0.0, 'position_y'),
                                                    ('PAD',)]
        object_selection_natural_sentence = object_selection_inference.tokenizer.convert_to_natural_sentence(object_selection_structured_sentence)
        structure_specification_natural_sentence = object_selection_inference.tokenizer.convert_structure_params_to_natural_language(structure_specification_structured_sentence)

        # object selection
        if args.random_select:
            num_obj = len(init_datum["object_pad_mask"]) - np.sum(init_datum["object_pad_mask"])
            predictions = np.random.choice([0, 1], num_obj)
            gts = np.ones(num_obj)
        else:
            predictions, gts = object_selection_inference.predict_target_objects(init_datum)

        all_obj_xyzs = init_datum["xyzs"][:len(predictions)]
        all_obj_rgbs = init_datum["rgbs"][:len(predictions)]
        obj_idxs = [i for i, l in enumerate(predictions) if l == 1.0]
        if len(obj_idxs) == 0:
            continue
        other_obj_idxs = [i for i, l in enumerate(predictions) if l == 0.0]
        obj_xyzs = [all_obj_xyzs[i] for i in obj_idxs]
        obj_rgbs = [all_obj_rgbs[i] for i in obj_idxs]
        other_obj_xyzs = [all_obj_xyzs[i] for i in other_obj_idxs]
        other_obj_rgbs = [all_obj_rgbs[i] for i in other_obj_idxs]

        print("\nSelect objects to rearrange...")
        print("Instruction:", object_selection_natural_sentence)
        if not args.gui_off:
            print("Visualize groundtruth (dot color) and prediction (ring color)")
            show_pcs_with_predictions(init_datum["xyzs"][:len(predictions)], init_datum["rgbs"][:len(predictions)],
                                    gts, predictions, add_table=True, side_view=True)
            print("Visualize object to rearrange")
            show_pcs(obj_xyzs, obj_rgbs, side_view=True, add_table=True)

        # pose generation
        max_num_objects = pose_generation_inference.cfg.dataset.max_num_objects
        max_num_other_objects = pose_generation_inference.cfg.dataset.max_num_other_objects
        if len(obj_xyzs) > max_num_objects:
            print("WARNING: reducing the number of \"query\" objects because this model is trained with a maximum of {} \"query\" objects. Train a new model if a larger number is needed.".format(max_num_objects))
            obj_xyzs = obj_xyzs[:max_num_objects]
            obj_rgbs = obj_rgbs[:max_num_objects]
        if len(other_obj_xyzs) > max_num_other_objects:
            print("WARNING: reducing the number of \"distractor\" objects because this model is trained with a maximum of {} \"distractor\" objects. Train a new model if a larger number is needed.".format(max_num_other_objects))
            other_obj_xyzs = other_obj_xyzs[:max_num_other_objects]
            other_obj_rgbs = other_obj_rgbs[:max_num_other_objects]

        pose_generation_datum = pose_generation_inference.dataset.prepare_test_data(obj_xyzs, obj_rgbs,
                                                                                    other_obj_xyzs, other_obj_rgbs,
                                                                                    {'length': structure_param['length'],
                                                                                    'length_increment': 0.05, 
                                                                                    'max_length': 1.0, 
                                                                                    'min_length': 0.0, 
                                                                                    'place_at_once': 'False', 
                                                                                    'position': [structure_param['position'], 0.0, 0.0], 
                                                                                    'rotation': [0.0, -0.0, 0.0], 
                                                                                    'type': 'dinner', 
                                                                                    'uniform_space': 'False'})
        datum = copy.deepcopy(pose_generation_datum)
        beam_pc_rearrangement = PointCloudRearrangement(datum)

        # autoregressive decoding
        num_target_objects = beam_pc_rearrangement.num_target_objects

        # first predict structure pose
        beam_goal_struct_pose, target_object_preds = pose_generation_inference.limited_batch_inference([datum])
        
        datum["struct_x_inputs"] = [beam_goal_struct_pose[0][0]]
        datum["struct_y_inputs"] = [beam_goal_struct_pose[0][1]]
        datum["struct_z_inputs"] = [beam_goal_struct_pose[0][2]]
        datum["struct_theta_inputs"] = [beam_goal_struct_pose[0][3:]]

        # then iteratively predict pose of each object
        struct_preds, target_object_preds = pose_generation_inference.limited_batch_inference([datum])
        # goal_obj_poses = []
        for obj_idx in range(num_target_objects):
            # goal_obj_poses.append(target_object_preds[0, obj_idx])
            datum["obj_x_inputs"][obj_idx] = target_object_preds[0][obj_idx][0]
            datum["obj_y_inputs"][obj_idx] = target_object_preds[0][obj_idx][1]
            datum["obj_z_inputs"][obj_idx] = target_object_preds[0][obj_idx][2]
            datum["obj_theta_inputs"][obj_idx] = target_object_preds[0][obj_idx][3:]
        # # concat in the object dim
        # beam_goal_obj_poses = np.stack(beam_goal_obj_poses, axis=0)
        # # swap axis
        # beam_goal_obj_poses = np.swapaxes(beam_goal_obj_poses, 1, 0)  # batch size, number of target objects, pose dim

        # move pc
        print('beam goal struct pose:')
        print(beam_goal_struct_pose[0])
        print('target object preds:')
        print(target_object_preds[0])
        struct_pose = beam_goal_struct_pose[0][:3]

        beam_pc_rearrangement.set_goal_poses(beam_goal_struct_pose[0], target_object_preds[0])
        beam_pc_rearrangement.rearrange()

        print("\nRearrange \"query\" objects...")
        print("Instruction:", structure_specification_natural_sentence)
        
        if not args.gui_off:
            print("Visualize rearranged scene sample")
            beam_pc_rearrangement.visualize("goal", add_other_objects=True, add_table=True, side_view=True)

        # then iteratively predict pose of each object
        # struct_preds, target_object_preds = pose_generation_inference.limited_batch_inference(beam_data)
        for obj_idx in range(num_target_objects):
            new_pose = beam_pc_rearrangement.goal_xyzs["xyzs"][obj_idx].mean(0).numpy()
            new_pose_delta = new_pose - struct_pose
            #print("new pose delta:")
            #print(new_pose_delta)
            scale = 1.0
            new_pose_delta = scale * new_pose_delta
            new_pose_delta = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ new_pose_delta
            translation = new_pose_delta + struct_pose - beam_pc_rearrangement.initial_xyzs["xyzs"][obj_idx].mean(0).numpy()
            #translation = beam_pc_rearrangement.goal_xyzs["xyzs"][obj_idx].mean(0).numpy()\
            #            - beam_pc_rearrangement.initial_xyzs["xyzs"][obj_idx].mean(0).numpy()

            #ratio = 1.0 #init_datum["depth"].max() / env.camera.origin_depth.max()
            #translation = translation * ratio
            # translation[2] = 0

            pid = env.pre_selected_objects[init_datum["shuffle_indices"][obj_idxs[obj_idx]]]
            orig_pos, orig_rot = p.getBasePositionAndOrientation(pid)

            rot = mat2quat(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ np.array(beam_pc_rearrangement.goal_poses["obj_poses"][obj_idx][3:]).reshape(3,3))
            #rot = mat2quat(np.array(beam_pc_rearrangement.goal_poses["obj_poses"][obj_idx][3:]).reshape(3,3))
            new_rot = quaternion_multiply(rot, orig_rot)

            # print(orig_pos)
            p.resetBasePositionAndOrientation(pid, orig_pos + translation, new_rot)
            for _ in range(100):
                p.stepSimulation()
            env.nvisii_update()

            obs = env.get_observation()
            currentRgb = obs[args.view]['rgb']
            currentSeg = obs[args.view]['segmentation']
            currentRgbFront = obs['front']['rgb']
            currentRgbNV = obs['nv-'+args.view]['rgb']
            currentSegNV = obs['nv-'+args.view]['segmentation']
            currentRgbFrontNV = obs['nv-front']['rgb']
            rgb_nobg = currentRgbNV[:, :, :3] * (currentSegNV!=1)[:, :, None]

            if args.get_reward:
                reward = getReward(rgb_nobg, rewardNet, preprocess)
                print("Current Score: %f" %reward)
                print("--------------------------------")

        if args.logging:
            cv2.imwrite('%s-%s/scene-%d/top_final.png'%(log_dir, log_name, sidx), cv2.cvtColor(currentRgbNV, cv2.COLOR_RGB2BGR))
            cv2.imwrite('%s-%s/scene-%d/front_final.png'%(log_dir, log_name, sidx), cv2.cvtColor(currentRgbFrontNV, cv2.COLOR_RGB2BGR))

            logger.info("-------------------------------")
            if args.get_reward:
                logger.info("Score: %f" %reward)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--dataset_base_dir", help='location of the dataset', type=str, 
                            default='/home/gun/ssd/disk/StructFormer/data_new_objects_test_split')
    parser.add_argument("--object_selection_model_dir", help='location for the saved object selection model', type=str,
                            default='/home/gun/Desktop/StructFormer/models/object_selection_network/best_model')
    parser.add_argument("--pose_generation_model_dir", help='location for the saved pose generation model', type=str,
                            default='/home/gun/Desktop/StructFormer/models/structformer_dinner/best_model')
    parser.add_argument("--dirs_config", help='config yaml file for directories', type=str,
                            default='/home/gun/Desktop/StructFormer/configs/data/dinner_dirs.yaml')
    parser.add_argument('--reward-model-path', type=str, default='../mcts/data/classification-best/top_nobg_linspace_mse-best.pth')
    # Environment settings
    parser.add_argument('--data-dir', type=str, default='/ssd/disk')
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument('--view', type=str, default='top') # 'front' / 'top'
    parser.add_argument('--use-template', action="store_true")
    parser.add_argument('--scenes', type=str, default='')# 'D1,D2,D3,D4,D5')
    parser.add_argument('--inorder', action="store_true")
    parser.add_argument('--random-select', action="store_true")
    parser.add_argument('--scene-split', type=str, default='all') # 'all' / 'seen' / 'unseen'
    parser.add_argument('--object-split', type=str, default='seen') # 'seen' / 'unseen'
    parser.add_argument('--num-objects', type=int, default=5)
    parser.add_argument('--num-scenes', type=int, default=10)
    parser.add_argument('--gui-off', action="store_true")
    parser.add_argument('--logging', action="store_true")
    parser.add_argument('--get-reward', action="store_true")
    args = parser.parse_args()

    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")
    os.environ["STRUCTFORMER"] = "/home/gun/Desktop/StructFormer"
    # # debug only
    # args.dataset_base_dir = "/home/weiyu/data_drive/data_new_objects_test_split"
    # args.object_selection_model_dir = "/home/weiyu/Research/intern/StructFormer/models/object_selection_network/best_model"
    # args.pose_generation_model_dir = "/home/weiyu/Research/intern/StructFormer/models/structformer_circle/best_model"
    # args.dirs_config = "/home/weiyu/Research/intern/StructFormer/configs/data/circle_dirs.yaml"

    if args.dirs_config:
        assert os.path.exists(args.dirs_config), "Cannot find config yaml file at {}".format(args.dirs_config)
        dirs_cfg = OmegaConf.load(args.dirs_config)
        dirs_cfg.dataset_base_dir = args.dataset_base_dir
        OmegaConf.resolve(dirs_cfg)
    else:
        dirs_cfg = None

    run_demo(args.object_selection_model_dir, args.pose_generation_model_dir, dirs_cfg, args)
