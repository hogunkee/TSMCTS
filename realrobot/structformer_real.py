import copy
import time
import torch
import numpy as np
import os
import argparse
import pyrealsense2 as rs
import logging
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
import pybullet as p
from matplotlib import pyplot as plt
from utils_pc import depth2pointcloud, get_raw_data, rotate_around_point
from transform_utils import mat2quat, quat2mat, mat2euler, euler2quat

from environment import RealEnvironment

# tabletop environment
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp/'))#pybullet_ur5_robotiq'))
from scene_utils import quaternion_multiply
#from custom_env import get_contact_objects, quaternion_multiply
#sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp'))
#from collect_template_list import scene_list
#sys.path.append(os.path.join(FILE_PATH, '..', 'mcts'))
#from utils import suppress_stdout
from mcts_real import Renderer, loadRewardFunction


class Discriminator(object):
    def __init__(self, renderer, args):
        self.renderer = renderer

        self.thresholdSuccess = args.threshold_success

        self.gtRewardNet = None

        self.batchSize = args.batch_size #32
        self.preProcess = None
    
    def reset(self, rgbImage, segmentation, classes=None):
        table = self.renderer.setup(rgbImage, segmentation, numRotations=4, classes=classes)
        return table

    def setGTRewardNet(self, gtRewardNet):
        self.gtRewardNet = gtRewardNet

    def setPreProcess(self, preProcess):
        self.preProcess = preProcess

    def getReward(self, tables, groundTruth=False):
        # print('getReward.')
        states = []
        for table in tables:
            rgb = self.renderer.getRGB(table)
            states.append(rgb)
        s_value = torch.Tensor(np.array(states)/255.).to(torch.float32).cuda()
        s_reward = torch.Tensor(np.array(states)/255.).permute([0,3,1,2]).cuda()
        if self.preProcess is not None:
            s_reward = self.preProcess(s_reward)

        if len(states) > self.batchSize:
            rewards = []
            numBatches = len(states)//self.batchSize
            if len(states)%self.batchSize > 0:
                numBatches += 1
            for b in range(numBatches):
                batchStatesR = s_reward[b*self.batchSize:(b+1)*self.batchSize]
                batchStatesV = [s_value[b*self.batchSize:(b+1)*self.batchSize], None, None]
                if groundTruth or self.rewardType=='gt':
                    batchRewards = self.gtRewardNet(batchStatesR).cpu().detach().numpy()
                else:
                    batchRewards = self.rewardNet(batchStatesV).cpu().detach().numpy()
                rewards.append(batchRewards)
            rewards = np.concatenate(rewards)
        else:
            if groundTruth or self.rewardType=='gt':
                rewards = self.gtRewardNet(s_reward).cpu().detach().numpy()
            else:
                rewards = self.rewardNet([s_value, None, None]).cpu().detach().numpy()
        return rewards.reshape(-1), [0.0]

    def isTerminal(self, node, table=None, checkReward=False, groundTruth=False):
        # print('isTerminal')
        terminal = False
        reward = 0.0
        value = 0.0
        if table is None:
            table = node.table
        # check collision and reward
        collision = self.renderer.checkCollision(table)
        if collision:
            reward = 0.0
            value = 0.0
            terminal = True
        elif checkReward:
            reward, value = self.getReward([table], groundTruth=groundTruth)
            reward, value = reward[0], value[0]
            if reward > self.thresholdSuccess:
                terminal = True
        return terminal, reward, value


def run_demo(object_selection_model_dir, pose_generation_model_dir, dirs_config, args):
    """
    Run a simple demo. Creates the object selection inference model, pose generation model,
    and so on. Requires paths to the model + config directories.
    """
    # Logger
    now = datetime.datetime.now()
    log_name = now.strftime("%m%d_%H%M")
    # logname = 'SF-' + log_name

    # Random seed
    seed = args.seed
    if seed is not None:
        print("Random seed: %d"%seed)
        random.seed(seed)
        np.random.seed(seed)

    # Environment setup
    env = RealEnvironment(None)

    # MCTS renderer 
    renderer = Renderer(tableSize=(args.H, args.W), imageSize=(360, 480), cropSize=(args.crop_size, args.crop_size))
    searcher = Discriminator(renderer, args) #1/np.sqrt(2)
    # Network setup
    model_path = args.reward_model_path
    gtRewardNet, preprocess = loadRewardFunction(model_path)
    searcher.setGTRewardNet(gtRewardNet)
    searcher.setPreProcess(preprocess)

    # StructFormer
    object_selection_inference = ObjectSelectionInference(object_selection_model_dir, dirs_cfg)
    pose_generation_inference = PriorInference(pose_generation_model_dir, dirs_cfg)

    success = 0
    success_eplen = []
    best_scores = []
    log_dir = 'data/SF'
    bar = range(1)
    for sidx in bar:
        while True:
            best_score = 0.0
            bestRgb = None
            if args.logging: 
                # bar.set_description("Episode %d/%d"%(sidx, args.num_scenes))
                os.makedirs('%s-%s/scene-%d'%(log_dir, log_name, sidx), exist_ok=True)
                with open('%s-%s/config.json'%(log_dir, log_name), 'w') as f:
                    json.dump(args.__dict__, f, indent=2)

                logger.handlers.clear()
                formatter = logging.Formatter('%(asctime)s - %(name)s -\n%(message)s')
                file_handler = logging.FileHandler('%s-%s/scene-%d/sflog.log'%(log_dir, log_name, sidx))
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            if seed is not None: 
                random.seed(seed + sidx)
                np.random.seed(seed + sidx)
            
            # Initial state
            #classes = ['fork', 'knife', 'plate', 'cup'] 
            classes = args.classes.replace(" ", "").replace(",", ".").split(".")

            while True:
                env.get_observation(move_ur5=True)
                obs = env.reset(classes, move_ur5=False, sort=True)
                T_rs = env.UR5.get_T_robot_to_rs()

                initRgb = obs['rgb_raw']
                initDepth = obs['depth_raw']
                initSeg = obs['segmentation_raw']
                initClasses = [classes[cid].lower() for cid in obs['class_id']]

                initDepth[initDepth==0] = np.mean(initDepth)

                plt.imshow(initRgb)
                plt.show()
                plt.imshow(initSeg)
                plt.show()
                x = input("Press 'r' for reset the table and get a new initial state.")
                if x.lower()!='r':
                    break

            table = searcher.reset(obs['rgb'], obs['segmentation'], initClasses)
            _, reward, _ = searcher.isTerminal(None, table, checkReward=True, groundTruth=True)
            print_fn("Score: %f"%reward)

            if args.logging:
                plt.imshow(obs['rgb'])
                plt.savefig('%s-%s/scene-%d/initial.png'%(log_dir, log_name, sidx))
            
            structure_param = {'length': np.random.uniform(0.25, 0.4),
                            'position': np.random.uniform(0.35, 0.65)}

            print_fn("--------------------------------")
            step = 0
            rgb, depth, seg = initRgb, initDepth, initSeg
            env.mapping_objs = env.data['object_names']
            while step<10:
                # from dataset
                xyz, rgb = depth2pointcloud(depth, rgb, env.RS.K_rs, T_rs)
                init_datum = get_raw_data(rgb, xyz, depth, seg, structure_param, max_num_objects=10)

                # test_datum = test_dataset.get_raw_data(idx)
                # goal_specification = init_datum["goal_specification"]
                # xyzs = init_datum["xyzs"] + test_datum["xyzs"]
                # rgbs = init_datum["rgbs"] + test_datum["rgbs"]
                # show_pcs(xyzs, rgbs, side_view=True, add_table=True)
                object_selection_structured_sentence = [('dinner', 'scene'), ('PAD',), ('PAD',), ('PAD',)]
                structure_specification_structured_sentence = [
                                                    ('dinner', 'shape'),
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

                print('Predictions:', predictions)

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
                if args.visualize:
                    print("Visualize groundtruth (dot color) and prediction (ring color)")
                    show_pcs_with_predictions(
                            init_datum["xyzs"][:len(predictions)], 
                            init_datum["rgbs"][:len(predictions)], 
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

                pose_generation_datum = pose_generation_inference.dataset.prepare_test_data(
                                            obj_xyzs, obj_rgbs, other_obj_xyzs, other_obj_rgbs,
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
                beam_pc_rearrangement.set_goal_poses(beam_goal_struct_pose[0], target_object_preds[0])
                beam_pc_rearrangement.rearrange()

                print("\nRearrange \"query\" objects...")
                print("Instruction:", structure_specification_natural_sentence)
                
                if args.visualize:
                    print("Visualize rearranged scene sample")
                    count = len([f for f in os.listdir('outputs/') if f.startswith('sample_')])
                    beam_pc_rearrangement.visualize("goal", add_other_objects=True, add_table=True, side_view=False, show_vis=True, save_vis=False, save_filename='outputs/sample_%d.png'%count)
                    #beam_pc_rearrangement.visualize("goal", add_other_objects=True, add_table=True, side_view=False, show_vis=False, save_vis=True, save_filename='outputs/sample_%d.png'%count)
                    #beam_pc_rearrangement.visualize("goal", add_other_objects=True, add_table=True, side_view=True)

                key = input('ok?')
                if key == 'y':
                    break
                
            # then iteratively predict pose of each object
            # struct_preds, target_object_preds = pose_generation_inference.limited_batch_inference(beam_data)
            for obj_idx in range(num_target_objects):
                # target_pose = target_object_preds[0][obj_idx]
                # initial_pose = beam_pc_rearrangements[0].initial_xyzs["xyzs"][obj_idx].mean(0).numpy()
                # translation = target_pose[:3] - initial_pose
                translation = beam_pc_rearrangement.goal_xyzs["xyzs"][obj_idx].mean(0).numpy()\
                            - beam_pc_rearrangement.initial_xyzs["xyzs"][obj_idx].mean(0).numpy()

                ratio = 1.0 #init_datum["depth"].max() / env.camera.origin_depth.max()
                translation = translation * ratio
                # translation[2] = 0

                rot_euler = mat2euler(np.array(beam_pc_rearrangement.goal_poses["obj_poses"][obj_idx][3:]).reshape(3,3))
                roll, pitch, yaw = rot_euler
                x,y,z,w = euler2quat([0, 0, yaw])
                rot_4dof = quat2mat([x,y,z,w])
                print(init_datum['shuffle_indices'][obj_idxs[obj_idx]] + 1)
                print('initial sf xyzs : ', beam_pc_rearrangement.initial_xyzs["xyzs"][obj_idx].mean(0).numpy())
                print('translation: ', translation)
                print('rot:')
                print(rot_euler)

                ## Rotate ##
                angle_delta = np.pi/2
                translation = rotate_around_point(translation.reshape(1, 3), angle_delta, (0, 0))[0]
                #yaw += angle_delta

                print('rotated translation:', translation)
                obs = env.step_3d(init_datum['shuffle_indices'][obj_idxs[obj_idx]] + 1 , translation, yaw, stop=False)

                if args.logging:
                    currentRgb = obs['rgb']
                    #currentDepth = obs['depth']
                    currentSeg = obs['segmentation']
                    currentClasses = [classes[cid].lower() for cid in obs['class_id']]

                    table = searcher.reset(currentRgb, currentSeg, currentClasses)
                    if table is None:
                        score = 0.
                    else:
                        _, reward, _ = searcher.isTerminal(None, table, checkReward=True, groundTruth=True)
                    print_fn("Score: %f"%reward)
                    plt.imshow(currentRgb)
                    plt.savefig('%s-%s/scene-%d/real_%d.png'%(log_dir, log_name, sidx, step)) 

                step += 1
                if step >= 10:
                    break

        print_fn("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--dataset_base_dir", help='location of the dataset', type=str,
                            default='/home/ur-plusle/Desktop/StructFormer/data_new_objects_test_split')

    parser.add_argument("--object_selection_model_dir", help='location for the saved object selection model', type=str,
                            default='/home/ur-plusle/Desktop/StructFormer/models/object_selection_network/best_model')
    parser.add_argument("--pose_generation_model_dir", help='location for the saved pose generation model', type=str,
                            default='/home/ur-plusle/Desktop/StructFormer/models/structformer_dinner/best_model')
    parser.add_argument("--dirs_config", help='config yaml file for directories', type=str,
                            default='/home/ur-plusle/Desktop/StructFormer/configs/data/dinner_dirs.yaml')
    # Environment settings
    #parser.add_argument('--data-dir', type=str, default='/ssd/disk')
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument('--use-template', action="store_true")
    parser.add_argument('--scenes', type=str, default='D1,D2,D3,D4,D5')
    parser.add_argument('--inorder', action="store_true")
    parser.add_argument('--random-select', action="store_true")
    parser.add_argument('--scene-split', type=str, default='all') # 'all' / 'seen' / 'unseen'
    parser.add_argument('--object-split', type=str, default='seen') # 'seen' / 'unseen'
    parser.add_argument('--num-objects', type=int, default=5)
    parser.add_argument('--num-scenes', type=int, default=16)
    parser.add_argument('--visualize', action="store_true")
    parser.add_argument('--logging', action="store_true")
    parser.add_argument('--classes', type=str, default="Fork.Knife.Plate.Bowl")

    parser.add_argument('--reward-model-path', type=str, default='../mcts/data/classification-best/top_nobg_linspace_mse-best.pth')
    parser.add_argument('--H', type=int, default=12)
    parser.add_argument('--W', type=int, default=15)
    parser.add_argument('--crop-size', type=int, default=192) #96 #128
    parser.add_argument('--threshold-success', type=float, default=0.95) #0.85
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    if args.logging:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    def print_fn(s=''):
        if args.logging: 
            logger.info(s)
            print(s)
        else: 
            print(s)

    #classes = args.classes.replace(" ", "").replace(",", ".").split(".")
    #args.classes = "Spoon.Fork.Knife.Glass.Cup.Bowl.Basket.Plate.Teapot.Shampoo.Clock.Toothpaste.Tube.Box.Marker.Stapler.Vaseline.Pen.Apple.Orange.Scissors.Box"

    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")
    os.environ["STRUCTFORMER"] = "/home/ur-plusle/Desktop/StructFormer"
    # # debug only
    # args.dataset_base_dir = "/home/weiyu/data_drive/data_new_objects_test_split"
    # args.object_selection_model_dir = "/home/weiyu/Research/intern/StructFormer/models/object_selection_network/best_model"
    # args.pose_generation_model_dir = "/home/weiyu/Research/intern/StructFormer/models/structformer_circle/best_model"
    # args.dirs_config = "/home/weiyu/Research/intern/StructFormer/configs/data/circle_dirs.yaml"

    #dirs_cfg = None
    if args.dirs_config:
        assert os.path.exists(args.dirs_config), "Cannot find config yaml file at {}".format(args.dirs_config)
        dirs_cfg = OmegaConf.load(args.dirs_config)
        dirs_cfg.dataset_base_dir = args.dataset_base_dir
        OmegaConf.resolve(dirs_cfg)
    else:
        dirs_cfg = None
    
    run_demo(args.object_selection_model_dir, args.pose_generation_model_dir, dirs_cfg, args)
