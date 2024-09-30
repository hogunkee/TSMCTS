import copy
import time
import torch
import numpy as np
import os
import argparse
import pyrealsense2 as rs
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
from utils_pc import depth2pointcloud, get_raw_data #, setupEnvironment
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

    object_selection_inference = ObjectSelectionInference(object_selection_model_dir, dirs_cfg)
    pose_generation_inference = PriorInference(pose_generation_model_dir, dirs_cfg)

    success = 0
    success_eplen = []
    best_scores = []
    log_dir = 'data/SF'
    bar = range(args.num_scenes)
    for sidx in bar:
        best_score = 0.0
        bestRgb = None
        if args.logging: 
            # bar.set_description("Episode %d/%d"%(sidx, args.num_scenes))
            os.makedirs('%s-%s/scene-%d'%(log_dir, log_name, sidx), exist_ok=True)
            with open('%s-%s/config.json'%(log_dir, log_name), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

        if seed is not None: 
            random.seed(seed + sidx)
            np.random.seed(seed + sidx)
        
        # Initial state
        classes = args.classes.replace(" ", "").replace(",", ".").split(".")

        obs = None
        count = 1
        while True:
            if count > 10:
                exit()
            if obs is not None:
                break
            try:
                obs = env.reset(classes)
                T_rs = env.UR5.get_T_robot_to_rs()
            except:
                print('count:', count)
                count += 1
                try:
                    env.RS.config_pipe()
                    env.RS.pipeline.start(env.RS.config)
                    env.RS.pipeline.wait_for_frames()
                    env.RS.pipeline.stop()
                except:
                    pass
        initRgb = obs['rgb_raw']
        initDepth = obs['depth_raw']
        initSeg = obs['segmentation_raw']
        plt.imshow(initSeg)
        plt.show()

        if args.logging:
            plt.imshow(initRgb)
            plt.savefig('%s-%s/scene-%d/initial.png'%(log_dir, log_name, sidx))
        
        structure_param = {'length': np.random.uniform(0.25, 0.4),
                           'position': np.random.uniform(0.35, 0.65)}

        print("--------------------------------")
        step = 0
        rgb, depth, seg = initRgb, initDepth, initSeg
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
                beam_pc_rearrangement.visualize("goal", add_other_objects=True, add_table=True, side_view=False, show_vis=False, save_vis=True, save_filename='outputs/sample_%d.png'%count)
                #beam_pc_rearrangement.visualize("goal", add_other_objects=True, add_table=True, side_view=True)

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
                print('rot:')
                print(rot_euler)

                if False:
                    pid = env.pre_selected_objects[init_datum["shuffle_indices"][obj_idxs[obj_idx]]]
                    orig_pos, orig_rot = p.getBasePositionAndOrientation(pid)


                    # print(orig_pos)
                    p.resetBasePositionAndOrientation(pid, orig_pos + translation, new_rot)
                    p.stepSimulation()

                    if args.logging:
                        obs = env.get_observation()

                        obs = env.step(target_object, target_position, rot_angle, stop=True)#False)
                        currentRgb = obs['rgb_raw']
                        currentDepth = obs['depth_raw']
                        currentSeg = obs['segmentation_raw']
                        plt.imshow(currentRgb)
                        plt.savefig('%s-%s/scene-%d/real_%d.png'%(log_dir, log_name, sidx, step)) 
                step += 1
                if step >= 10:
                    break

        print("Done.")


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
    parser.add_argument('--classes', type=str, default="Apple. Lemon. Orange. Fruit.")
    args = parser.parse_args()

    #classes = args.classes.replace(" ", "").replace(",", ".").split(".")
    args.classes = "Spoon.Fork.Knife.Glass.Cup.Bowl.Basket.Plate.Teapot.Shampoo.Clock.Toothpaste.Tube.Box.Marker.Stapler.Vaseline.Pen.Apple.Orange.Scissors.Box"

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
