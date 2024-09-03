import os
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf

from StructDiffusion.data.semantic_arrangement import SemanticArrangementDataset
from StructDiffusion.language.tokenizer import Tokenizer
from StructDiffusion.models.pl_models import ConditionalPoseDiffusionModel, PairwiseCollisionModel
from StructDiffusion.diffusion.sampler import SamplerV2
from StructDiffusion.diffusion.pose_conversion import get_struct_objs_poses
from StructDiffusion.utils.files import get_checkpoint_path_from_dir, replace_config_for_testing_data
from StructDiffusion.utils.batch_inference import move_pc_and_create_scene_simple, visualize_batch_pcs

import json
import datetime
import random
import pybullet as p
from matplotlib import pyplot as plt
from utils_pc import depth2pointcloud, get_diffusion_data #, setupEnvironment
from transform_utils import mat2quat

from environment import RealEnvironment

# tabletop environment
#import sys
#FILE_PATH = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp/pybullet_ur5_robotiq'))
#from custom_env import get_contact_objects, quaternion_multiply
#sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp'))
#from collect_template_list import scene_list
#sys.path.append(os.path.join(FILE_PATH, '..', 'mcts'))
#from utils import suppress_stdout


def main(args, cfg):
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
        pl.seed_everything(args.seed)

    # Environment setup
    env = RealEnvironment(None)

    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    diffusion_checkpoint_dir = os.path.join(cfg.WANDB.save_dir, cfg.WANDB.project, args.diffusion_checkpoint_id, "checkpoints")
    diffusion_checkpoint_path = get_checkpoint_path_from_dir(diffusion_checkpoint_dir)
    collision_checkpoint_dir = os.path.join(cfg.WANDB.save_dir, cfg.WANDB.project, args.collision_checkpoint_id, "checkpoints")
    collision_checkpoint_path = get_checkpoint_path_from_dir(collision_checkpoint_dir)

    tokenizer = Tokenizer(cfg.DATASET.vocab_dir)
    # override ignore_rgb for visualization
    cfg.DATASET.ignore_rgb = False
    dataset = SemanticArrangementDataset(split="test", tokenizer=tokenizer, **cfg.DATASET)

    sampler = Sampler(ConditionalPoseDiffusionModel, diffusion_checkpoint_path, 
                      PairwiseCollisionModel, collision_checkpoint_path, device)
    #sampler = Sampler(ConditionalPoseDiffusionModel, checkpoint_path, device)
    
    log_dir = 'data/SD'
    bar = range(args.num_scenes)
    for sidx in bar:
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

        initRgb = obs['rgb_raw']
        initDepth = obs['depth_raw']
        initSeg = obs['segmentation_raw']

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
            if False:
                init_datum = dataset.get_raw_data(step)
                # for i in range(len(init_datum["goal_poses"])):
                #     init_datum["goal_poses"][i] = np.eye(4)
                # init_datum["t"] = 200
                print(tokenizer.convert_structure_params_to_natural_language(init_datum["sentence"]))
                datum = dataset.convert_to_tensors(init_datum, tokenizer)
                batch = dataset.single_datum_to_batch(datum, args.num_samples, device, inference_mode=True)
                num_poses = datum["goal_poses"].shape[0]
                #xs = sampler.sample(batch, num_poses) 
            # from pybullet env
            else:
                try:
                    xyz, rgb = depth2pointcloud(depth, rgb, env.RS.K_rs, T_rs)
                    init_datum = get_diffusion_data(rgb, xyz, depth, seg, structure_param, inference_mode=True)
                    #init_datum = get_diffusion_data(env.get_observation(), env, structure_param, view=args.view, inference_mode=True)
                except:
                    break
                print(tokenizer.convert_structure_params_to_natural_language(init_datum["sentence"]))
                obj_ids = init_datum["ids"]
                target_objects = init_datum["target_objs"]

                datum = dataset.convert_to_tensors(init_datum, tokenizer)
                batch = dataset.single_datum_to_batch(datum, args.num_samples, device, inference_mode=True)
                batch_copy = batch.copy()
                num_poses = datum["goal_poses"].shape[0]
                #xs = sampler.sample(batch, num_poses)

            struct_pose, pc_poses_in_struct = sampler.sample(batch, num_poses)
            #struct_pose, pc_poses_in_struct = get_struct_objs_poses(xs[0])
            # struct_pose : 10, 1, 4, 4
            # pc_poses_in_struct : 10, 7, 4, 4
            # dataset: [0.3891, 0.0556, 0.0293]
            # pybullet: [0.4332, 0.2187, 0.1527]
            new_obj_xyzs = move_pc_and_create_scene_simple(batch["pcs"], struct_pose, pc_poses_in_struct)
            if args.visualize:
                visualize_batch_pcs(torch.cat(init_datum["pcs"]).reshape(1, 7, 1024, 6), args.num_samples)
                visualize_batch_pcs(new_obj_xyzs, args.num_samples, limit_B=10, trimesh=True)

            struct_rot = struct_pose[0][0][:3, :3]
            
            # num_objects = num_poses - 1
            for obj_idx, tobj_name in enumerate(target_objects):

                translation = new_obj_xyzs[0][obj_idx].mean(0)[:3].cpu().numpy() - init_datum["pcs"][obj_idx].mean(0)[:3].numpy()
                object_rot = pc_poses_in_struct[0][obj_idx][:3, :3]
                rot = mat2quat(struct_rot.cpu().numpy() @ object_rot.cpu().numpy())
                new_rot = quaternion_multiply(rot, orig_rot)

                if False:
                    pid = obj_ids[tobj_name]
                    # pid = env.pre_selected_objects[obj_ids[tobj_name]]
                    # pid = env.pre_selected_objects[obj_idx] #init_datum["shuffle_indices"][obj_idxs[obj_idx]]]
                    orig_pos, orig_rot = p.getBasePositionAndOrientation(pid)

                    p.resetBasePositionAndOrientation(pid, orig_pos + translation, new_rot)
                    p.stepSimulation()
                step += 1

                if args.logging:
                    obs = env.get_observation()
                    currentRgb = obs['top']['rgb']
                    plt.imshow(currentRgb)
                    plt.savefig('%s-%s/scene-%d/real_%d.png'%(log_dir, log_name, sidx, step))
                if step >= 10:
                    break
            
        print("Done.")
        # # from dataset
        # raw_datum = dataset.get_raw_data(di)
        # print(tokenizer.convert_structure_params_to_natural_language(raw_datum["sentence"]))
        # datum = dataset.convert_to_tensors(raw_datum, tokenizer)
        # batch = dataset.single_datum_to_batch(datum, args.num_samples, device, inference_mode=True)
        # num_poses = datum["goal_poses"].shape[0]
        # xs = sampler.sample(batch, num_poses) 
        
        # struct_pose, pc_poses_in_struct = get_struct_objs_poses(xs[0])
        # new_obj_xyzs = move_pc_and_create_scene_simple(batch["pcs"], struct_pose, pc_poses_in_struct)
        # visualize_batch_pcs(new_obj_xyzs, args.num_samples, limit_B=10, trimesh=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer")
    parser.add_argument("--base_config_file", help='base config yaml file',
                        default='/home/ur-plusle/Desktop/StructDiffusion/configs/base.yaml',
                        type=str)
    parser.add_argument("--config_file", help='config yaml file',
                        default='/home/ur-plusle/Desktop/StructDiffusion/configs/conditional_pose_diffusion.yaml',
                        type=str)
    parser.add_argument("--testing_data_config_file", help='config yaml file',
                        default='/home/ur-plusle/Dekstop/StructDiffusion/configs/testing_data.yaml',
                        type=str)
    parser.add_argument("--diffusion-checkpoint_id",
                        default="ConditionalPoseDiffusion",
                        type=str)
    parser.add_argument("--collision-checkpoint_id",
                        default="CollisionDiscriminator",
                        type=str)
    parser.add_argument("--num_samples",
                        default=1, #10,
                        type=int)
    # Environment settings
    #parser.add_argument('--data-dir', type=str, default='/ssd/disk')
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument('--view', type=str, default='front') # 'front' / 'top'
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

    args.classes = "Spoon.Fork.Knife.Glass.Cup.Bowl.Basket.Plate.Teapot.Shampoo.Clock.Toothpaste.Tube.Box.Marker.Stapler.Vaseline.Pen.Apple.Orange.Scissors.Box"

    base_cfg = OmegaConf.load(args.base_config_file)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(base_cfg, cfg)

    testing_data_cfg = OmegaConf.load(args.testing_data_config_file)
    testing_data_cfg = OmegaConf.merge(base_cfg, testing_data_cfg)
    replace_config_for_testing_data(cfg, testing_data_cfg)

    main(args, cfg)


