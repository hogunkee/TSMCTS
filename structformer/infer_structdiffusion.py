import os
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf

from StructDiffusion.data.semantic_arrangement import SemanticArrangementDataset
from StructDiffusion.language.tokenizer import Tokenizer
from StructDiffusion.models.pl_models import ConditionalPoseDiffusionModel
from StructDiffusion.diffusion.sampler import Sampler
from StructDiffusion.diffusion.pose_conversion import get_struct_objs_poses
from StructDiffusion.utils.files import get_checkpoint_path_from_dir, replace_config_for_testing_data
from StructDiffusion.utils.batch_inference import move_pc_and_create_scene_simple, visualize_batch_pcs

import datetime
import random
import pybullet as p
from matplotlib import pyplot as plt
from pc_utils import get_diffusion_data, setupEnvironment
from transform_utils import mat2quat

# tabletop environment
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp/pybullet_ur5_robotiq'))
from custom_env import get_contact_objects, quaternion_multiply
sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp'))
from collect_template_list import scene_list
sys.path.append(os.path.join(FILE_PATH, '..', 'mcts'))
from utils import suppress_stdout


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


    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    checkpoint_dir = os.path.join(cfg.WANDB.save_dir, cfg.WANDB.project, args.checkpoint_id, "checkpoints")
    checkpoint_path = get_checkpoint_path_from_dir(checkpoint_dir)

    tokenizer = Tokenizer(cfg.DATASET.vocab_dir)
    # override ignore_rgb for visualization
    cfg.DATASET.ignore_rgb = False
    dataset = SemanticArrangementDataset(split="test", tokenizer=tokenizer, **cfg.DATASET)

    sampler = Sampler(ConditionalPoseDiffusionModel, checkpoint_path, device)

    data_idxs = np.random.permutation(len(dataset))
    for di in data_idxs:
        # from pybullet env
        env.arrange_objects(random=True)
        structure_param = {'length': np.random.uniform(0.25, 0.4),
                        'position': np.random.uniform(0.35, 0.65)}
        init_datum = get_diffusion_data(env.get_observation(), env, structure_param, view=args.view)
        datum2 = dataset.convert_to_tensors(init_datum, tokenizer)
        batch2 = dataset.single_datum_to_batch(datum2, args.num_samples, device, inference_mode=True)
        num_poses2 = datum2["goal_poses"].shape[0]
        xs2 = sampler.sample(batch2, num_poses2)
        
        # from dataset
        raw_datum = dataset.get_raw_data(di)
        print(tokenizer.convert_structure_params_to_natural_language(raw_datum["sentence"]))
        datum = dataset.convert_to_tensors(raw_datum, tokenizer)
        batch = dataset.single_datum_to_batch(datum, args.num_samples, device, inference_mode=True)
        num_poses = datum["goal_poses"].shape[0]
        xs = sampler.sample(batch, num_poses) 
        
        #TODO
        struct_pose, pc_poses_in_struct = get_struct_objs_poses(xs[0])
        new_obj_xyzs = move_pc_and_create_scene_simple(batch["pcs"], struct_pose, pc_poses_in_struct)
        visualize_batch_pcs(new_obj_xyzs, args.num_samples, limit_B=10, trimesh=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer")
    parser.add_argument("--base_config_file", help='base config yaml file',
                        default='../../StructDiffusion/configs/base.yaml',
                        type=str)
    parser.add_argument("--config_file", help='config yaml file',
                        default='../../StructDiffusion/configs/conditional_pose_diffusion.yaml',
                        type=str)
    parser.add_argument("--testing_data_config_file", help='config yaml file',
                        default='../../StructDiffusion/configs/testing_data.yaml',
                        type=str)
    parser.add_argument("--checkpoint_id",
                        default="ConditionalPoseDiffusion",
                        type=str)
    parser.add_argument("--num_samples",
                        default=10,
                        type=int)
    # Environment settings
    parser.add_argument('--data-dir', type=str, default='/ssd/disk')
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument('--view', type=str, default='front') # 'front' / 'top'
    parser.add_argument('--use-template', action="store_true")
    parser.add_argument('--scenes', type=str, default='D1,D2,D3,D4,D5')
    parser.add_argument('--inorder', action="store_true")
    parser.add_argument('--random-select', action="store_true")
    parser.add_argument('--scene-split', type=str, default='all') # 'all' / 'seen' / 'unseen'
    parser.add_argument('--object-split', type=str, default='seen') # 'seen' / 'unseen'
    parser.add_argument('--num-objects', type=int, default=5)
    parser.add_argument('--num-scenes', type=int, default=10)
    parser.add_argument('--gui-off', action="store_true")
    parser.add_argument('--logging', action="store_true")
    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.base_config_file)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(base_cfg, cfg)

    testing_data_cfg = OmegaConf.load(args.testing_data_config_file)
    testing_data_cfg = OmegaConf.merge(base_cfg, testing_data_cfg)
    replace_config_for_testing_data(cfg, testing_data_cfg)

    main(args, cfg)


