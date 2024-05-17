import copy
import time
import torch
import numpy as np
import os
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from structformer.data.tokenizer import Tokenizer
from structformer.evaluation.test_object_selection_network import ObjectSelectionInference
from structformer.evaluation.test_structformer import PriorInference
from structformer.utils.rearrangement import show_pcs_with_predictions, get_initial_scene_idxs, evaluate_target_object_predictions, save_img, show_pcs_with_labels, test_new_vis, show_pcs
from structformer.evaluation.inference import PointCloudRearrangement

# point cloud utils
from pc_utils import get_raw_data #depth2pc

import random
import pybullet as p
from matplotlib import pyplot as plt

# tabletop environment
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp/pybullet_ur5_robotiq'))
from custom_env import TableTopTidyingUpEnv, get_contact_objects, quaternion_multiply
from utilities import Camera, Camera_front_top
sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp'))
# from scene_utils import 
from collect_template_list import scene_list
sys.path.append(os.path.join(FILE_PATH, '..', 'mcts'))
from utils import suppress_stdout

from transform_utils import mat2quat


def setupEnvironment(args):
    camera_top = Camera((0, 0, 1.45), 0.02, 2, (480, 360), 60)
    camera_front_top = Camera_front_top((0.5, 0, 1.3), 0.02, 2, (480, 360), 60)
    
    data_dir = args.data_dir
    objects_cfg = { 'paths': {
            'pybullet_object_path' : os.path.join(data_dir, 'pybullet-URDF-models/urdf_models/models'),
            'ycb_object_path' : os.path.join(data_dir, 'YCB_dataset'),
            'housecat_object_path' : os.path.join(data_dir, 'housecat6d/obj_models_small_size_final'),
        },
        'split' : args.object_split #'inference' #'train'
    }
    
    gui_on = not args.gui_off
    env = TableTopTidyingUpEnv(objects_cfg, camera_top, camera_front_top, vis=gui_on, gripper_type='85')
    p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off
    p.addUserDebugLine([0, -0.5, 0], [0, -0.5, 1.1], [0, 1, 0])

    env.reset()
    return env

def run_demo(object_selection_model_dir, pose_generation_model_dir, dirs_config, args, beam_size=1):
    """
    Run a simple demo. Creates the object selection inference model, pose generation model,
    and so on. Requires paths to the model + config directories.
    """
    # Environment setup
    with suppress_stdout():
        env = setupEnvironment(args)
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

    test_dataset = object_selection_inference.dataset
    initial_scene_idxs = get_initial_scene_idxs(test_dataset)

    for idx in range(len(test_dataset)):
        if idx not in initial_scene_idxs:
            continue

        if idx == 4:
            continue

        filename, _ = test_dataset.get_data_index(idx)
        scene_id = os.path.split(filename)[1][4:-3]
        print("-"*50)
        print("Scene No.{}".format(scene_id))

        # retrieve data
        init_datum = get_raw_data(env.get_observation(), env)
        # test_datum = test_dataset.get_raw_data(idx)
        # goal_specification = init_datum["goal_specification"]
        # xyzs = init_datum["xyzs"] + test_datum["xyzs"]
        # rgbs = init_datum["rgbs"] + test_datum["rgbs"]
        # show_pcs(xyzs, rgbs, side_view=True, add_table=True)
        object_selection_structured_sentence = [('dinner', 'scene'), ('PAD',), ('PAD',), ('PAD',)]
        structure_specification_structured_sentence = [('dinner', 'shape'),
                                                    (0.0, 'rotation'),
                                                    (0.4856287214206586, 'position_x'),
                                                    (0.0, 'position_y'),
                                                    ('PAD',)]
        # object_selection_structured_sentence = init_datum["sentence"][5:]
        # structure_specification_structured_sentence = init_datum["sentence"][:5]
        # init_datum["sentence"] = object_selection_structured_sentence + structure_specification_structured_sentence
        object_selection_natural_sentence = object_selection_inference.tokenizer.convert_to_natural_sentence(object_selection_structured_sentence)
        structure_specification_natural_sentence = object_selection_inference.tokenizer.convert_structure_params_to_natural_language(structure_specification_structured_sentence)

        # object selection
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
                                                                                    {'length': 0.2631578947368421, 
                                                                                     'length_increment': 0.05, 
                                                                                     'max_length': 1.0, 
                                                                                     'min_length': 0.0, 
                                                                                     'place_at_once': 'False', 
                                                                                     'position': [0.4856287214206586, 0.0, 0.0], 
                                                                                     'rotation': [0.0, -0.0, 0.0], 
                                                                                     'type': 'dinner', 
                                                                                     'uniform_space': 'False'})
        beam_data = []
        beam_pc_rearrangements = []
        for b in range(beam_size):
            datum_copy = copy.deepcopy(pose_generation_datum)
            beam_data.append(datum_copy)
            beam_pc_rearrangements.append(PointCloudRearrangement(datum_copy))

        # autoregressive decoding
        num_target_objects = beam_pc_rearrangements[0].num_target_objects

        # first predict structure pose
        beam_goal_struct_pose, target_object_preds = pose_generation_inference.limited_batch_inference(beam_data)
        for b in range(beam_size):
            datum = beam_data[b]
            datum["struct_x_inputs"] = [beam_goal_struct_pose[b][0]]
            datum["struct_y_inputs"] = [beam_goal_struct_pose[b][1]]
            datum["struct_z_inputs"] = [beam_goal_struct_pose[b][2]]
            datum["struct_theta_inputs"] = [beam_goal_struct_pose[b][3:]]

        
        # then iteratively predict pose of each object
        beam_goal_obj_poses = []
        for obj_idx in range(num_target_objects):
            struct_preds, target_object_preds = pose_generation_inference.limited_batch_inference(beam_data)
            beam_goal_obj_poses.append(target_object_preds[:, obj_idx])
            for b in range(beam_size):
                datum = beam_data[b]
                datum["obj_x_inputs"][obj_idx] = target_object_preds[b][obj_idx][0]
                datum["obj_y_inputs"][obj_idx] = target_object_preds[b][obj_idx][1]
                datum["obj_z_inputs"][obj_idx] = target_object_preds[b][obj_idx][2]
                datum["obj_theta_inputs"][obj_idx] = target_object_preds[b][obj_idx][3:]
        # concat in the object dim
        beam_goal_obj_poses = np.stack(beam_goal_obj_poses, axis=0)
        # swap axis
        beam_goal_obj_poses = np.swapaxes(beam_goal_obj_poses, 1, 0)  # batch size, number of target objects, pose dim

        # move pc
        for bi in range(beam_size):
            beam_pc_rearrangements[bi].set_goal_poses(beam_goal_struct_pose[bi], beam_goal_obj_poses[bi])
            beam_pc_rearrangements[bi].rearrange()

        print("\nRearrange \"query\" objects...")
        print("Instruction:", structure_specification_natural_sentence)
        for pi, pc_rearrangement in enumerate(beam_pc_rearrangements):
            print("Visualize rearranged scene sample {}".format(pi))
            pc_rearrangement.visualize("goal", add_other_objects=True, add_table=True, side_view=True)


        
        # then iteratively predict pose of each object
        # struct_preds, target_object_preds = pose_generation_inference.limited_batch_inference(beam_data)
        for obj_idx in range(num_target_objects):
            # target_pose = target_object_preds[0][obj_idx]
            # initial_pose = beam_pc_rearrangements[0].initial_xyzs["xyzs"][obj_idx].mean(0).numpy()
            # translation = target_pose[:3] - initial_pose
            translation = beam_pc_rearrangements[0].goal_xyzs["xyzs"][obj_idx].mean(0).numpy()\
                         - beam_pc_rearrangements[0].initial_xyzs["xyzs"][obj_idx].mean(0).numpy()

            ratio = init_datum["depth"].max() / env.camera.origin_depth.max()
            translation = translation * ratio
            # translation[2] = 0

            
            pid = env.pre_selected_objects[obj_idxs[obj_idx]]
            orig_pos, orig_rot = p.getBasePositionAndOrientation(pid)

            rot = mat2quat(np.array(beam_pc_rearrangements[0].goal_poses["obj_poses"][obj_idx][3:]).reshape(3,3))
            new_rot = quaternion_multiply(rot, orig_rot)

            # print(orig_pos)
            p.resetBasePositionAndOrientation(pid, orig_pos + translation, new_rot)
            p.stepSimulation()

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
    # Environment settings
    parser.add_argument('--data-dir', type=str, default='/ssd/disk')
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument('--use-template', action="store_true")
    parser.add_argument('--scenes', type=str, default='D1,D2,D3,D4,D5')
    parser.add_argument('--inorder', action="store_true")
    parser.add_argument('--scene-split', type=str, default='all') # 'all' / 'seen' / 'unseen'
    parser.add_argument('--object-split', type=str, default='seen') # 'seen' / 'unseen'
    parser.add_argument('--num-objects', type=int, default=5)
    parser.add_argument('--num-scenes', type=int, default=10)
    parser.add_argument('--gui-off', action="store_true")
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
