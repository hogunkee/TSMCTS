import os
import sys
import numpy as np
import random
import torch

from structformer.utils.rearrangement import get_pts
import StructDiffusion.utils.transformations as tra

# tabletop environment
import sys
import pybullet as p

def depth2pointcloud(depth, rgb, K, T, depth_scale=1., clip_distance_max=1.):
    rows, cols  = depth.shape

    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    r = r.astype(float)
    c = c.astype(float)

    fx = K[0, 0]
    fy = K[1, 1]
    ppx = K[0, 2]
    ppy = K[1, 2]

    z = depth 
    x =  z * (c - ppx) / fx
    y =  z * (r - ppy) / fy
    print('x y z:', x.mean(), y.mean(), z.mean())

    #points_xyz = np.concatenate([x[:,:,None] ,y[:,:,None] ,z[:,:,None]], 2)
    xp = x.reshape(-1, 1).dot(T[:3, 0].reshape(1,3)).reshape(480,640,-1)
    yp = y.reshape(-1, 1).dot(T[:3, 1].reshape(1,3)).reshape(480,640,-1)
    zp = z.reshape(-1, 1).dot(T[:3, 2].reshape(1,3)).reshape(480,640,-1)
    tp = np.ones([480*640, 1]).dot(T[:3, 3].reshape(1,3)).reshape(480,640,-1)
    points_xyz = xp + yp + zp + tp
    points_rgb = rgb
    print("new x y z:", points_xyz[:,:,0].mean(), points_xyz[:,:,1].mean(), points_xyz[:,:,2].mean())
   
    return points_xyz, points_rgb

def depth2pc(depth, K, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    mask = np.where(depth > 0)
    x,y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]
        
    pc = np.vstack((world_x, world_y, world_z)).T
    return (pc, rgb)



#def get_raw_data(obs, env, structure_param, view='top', max_num_objects=10, num_pts=1024):
def get_raw_data(rgb, xyz, depth, seg, structure_param, max_num_objects=10, num_pts=1024):
    #rgb, depth, seg, valid, xyz = scene
    #rgb = obs[view]['rgb']
    #depth = obs[view]['depth']
    #seg = obs[view]['segmentation']

    #ids = {'table': 1}
    #for id, tobj in env.table_objects_list.items():
    #    ids[tobj[0]] = id
    #all_objs = [tobj[0] for id, tobj in env.table_objects_list.items()]
    #num_rearrange_objs = len(env.table_objects_list)
    num_rearrange_objs = int(seg.max())

    valid = np.logical_and(seg > 0, seg < 1000)

    # getting object point clouds
    obj_xyzs = []
    obj_rgbs = []
    object_pad_mask = []
    rearrange_obj_labels = []
    #for i, obj in enumerate(all_objs):
    for i in range(num_rearrange_objs):
        obj_mask = np.logical_and(seg==(i+1), valid)
        #obj_mask = np.logical_and(seg==ids[obj], valid)
        if np.sum(obj_mask) <= 0:
            raise Exception
        ok, obj_xyz, obj_rgb, _ = get_pts(xyz, rgb, obj_mask, num_pts=num_pts)
        obj_xyzs.append((obj_xyz + np.array([0.5, 0.5, 0.])).to(torch.float32))
        #obj_xyz = (obj_xyz - np.array([-0.5, 0, 0.58])).to(torch.float32) # [-0.475, 0, 0.58], [-0.475, 0, 0.42]
        #obj_xyzs.append((obj_xyz - np.array([-0.5, 0, 0.62])).to(torch.float32))
        obj_rgbs.append(obj_rgb[:, :3]/255)
        object_pad_mask.append(0)
        rearrange_obj_labels.append(1.0)

    # pad data
    for i in range(max_num_objects - num_rearrange_objs):
        obj_xyzs.append(torch.zeros([1024, 3], dtype=torch.float32))
        obj_rgbs.append(torch.zeros([1024, 3], dtype=torch.float32))
        rearrange_obj_labels.append(-100.0)
        object_pad_mask.append(1)

    ###################################
    # preparing sentence
    sentence = []
    sentence_pad_mask = []
    # structure parameters
    # 5 parameters
    structure_parameters = {'length': structure_param['length'], #0.2631578947368421,
                            'length_increment': 0.05,
                            'max_length': 1.0,
                            'min_length': 0.0,
                            'place_at_once': 'False',
                            'position': [structure_param['position'], 0.0, 0.0],
                            #'position': [0.4856287214206586, 0.0, 0.0],
                            'rotation': [0.0, -0.0, 0.0],
                            'type': 'dinner',
                            'uniform_space': 'False'}
    #goal_specification["shape"]
    sentence.append((structure_parameters["type"], "shape"))
    sentence.append((structure_parameters["rotation"][2], "rotation"))
    sentence.append((structure_parameters["position"][0], "position_x"))
    sentence.append((structure_parameters["position"][1], "position_y"))
    for _ in range(4):
        sentence_pad_mask.append(0)
    sentence.append(tuple(["PAD"]))
    sentence_pad_mask.append(1)

    # object selection
    is_anchor = False #len(goal_specification["anchor"]["features"]) > 0
    # rearrange
    for tf in [{'comparator': None, 'type': 'scene', 'value': 'dinner'}]: #goal_specification["rearrange"]["features"]:
        comparator = tf["comparator"]
        type = tf["type"]
        value = tf["value"]
        if comparator is None:
            # discrete features
            if is_anchor:
                # leave the desired value to be inferred from anchor
                sentence.append(("MASK", type))
            else:
                sentence.append((value, type))
        else:
            # continous features
            sentence.append((comparator, type))
        sentence_pad_mask.append(0)
    # pad, because we always have the fixed length, we don't need to pad this part of the sentence
    # assert len(goal_specification["rearrange"]["features"]) == self.max_num_rearrange_features

    # anchor
    for tf in []: #goal_specification["anchor"]["features"]:
        assert tf["comparator"] is None
        type = tf["type"]
        value = tf["value"]
        # discrete features
        sentence.append((value, type))
        sentence_pad_mask.append(0)
    # pad
    for i in range(3): #self.max_num_anchor_features): # - len(goal_specification["anchor"]["features"])):
        sentence.append(tuple(["PAD"]))
        sentence_pad_mask.append(1)

    # used to indicate whether the token is an object point cloud or a part of the instruction
    # assert self.max_num_rearrange_features + self.max_num_anchor_features + self.max_num_shape_parameters == len(sentence)
    assert max_num_objects == len(rearrange_obj_labels)
    token_type_index = [0] * len(sentence) + [1] * max_num_objects
    position_index = list(range(len(sentence))) + [i for i in range(max_num_objects)]

    # shuffle the position of objects since now the order is rearrange, anchor, distract
    shuffle_object_indices = list(range(num_rearrange_objs))
    random.shuffle(shuffle_object_indices)
    shuffle_object_indices = shuffle_object_indices + list(range(num_rearrange_objs, max_num_objects))
    obj_xyzs = [obj_xyzs[i] for i in shuffle_object_indices]
    obj_rgbs = [obj_rgbs[i] for i in shuffle_object_indices]
    object_pad_mask = [object_pad_mask[i] for i in shuffle_object_indices]
    rearrange_obj_labels = [rearrange_obj_labels[i] for i in shuffle_object_indices]

    datum = {
        "xyzs": obj_xyzs,
        "rgbs": obj_rgbs,
        "object_pad_mask": object_pad_mask,
        "rearrange_obj_labels": rearrange_obj_labels,
        "sentence": sentence,
        "sentence_pad_mask": sentence_pad_mask,
        "token_type_index": token_type_index,
        "position_index": position_index,
        "t": 0, #t,
        "filename": "", #filename,
        "goal_specification": None, #goal_specification,
        "depth": depth,
        "shuffle_indices": shuffle_object_indices[:num_rearrange_objs]
    }

    return datum

#def get_diffusion_data(obs, env, structure_param, view='top', inference_mode=False, \
#                        shuffle_object_index=False, num_pts=1024):
def get_diffusion_data(rgb, xyz, depth, seg, structure_param, inference_mode=False, \
                        shuffle_object_index=False, num_pts=1024):
    ignore_rgb = False #True
    use_virtual_structure_frame = True
    ignore_distractor_objects = True
    max_num_objects = 7
    max_num_shape_parameters = 5
    max_num_other_objects = 5
    structure_parameters = {'length': structure_param['length'], #0.2631578947368421,
                            'length_increment': 0.05,
                            'max_length': 1.0,
                            'min_length': 0.0,
                            'place_at_once': 'False',
                            'position': [structure_param['position'], 0.0, 0.0],
                            #'position': [0.4856287214206586, 0.0, 0.0],
                            'rotation': [0.0, -0.0, 0.0],
                            'type': 'dinner',
                            'uniform_space': 'False'}

    # getting scene images and point clouds
    #rgb, depth, seg, valid, xyz = scene

    num_rearrange_objs = int(seg.max())
    target_objs = np.arange(num_rearrange_objs)
    other_objs = []

    #ids = {'table': 1}
    #for id, tobj in env.table_objects_list.items():
    #    ids[tobj[0]] = id
    #all_objs = [tobj[0] for id, tobj in env.table_objects_list.items()]
    #num_rearrange_objs = len(env.table_objects_list)
    #target_objs = all_objs[:num_rearrange_objs]
    #other_objs = all_objs[num_rearrange_objs:]

    step_t = num_rearrange_objs
    valid = np.logical_and(seg > 0, seg < 1000)
    
    scene = rgb, depth, seg, valid, xyz
    if inference_mode:
        initial_scene = scene

    # getting object point clouds
    obj_pcs = []
    obj_pad_mask = []
    current_pc_poses = []
    other_obj_pcs = []
    other_obj_pad_mask = []
    #for obj in all_objs:
    for i in range(num_rearrange_objs):
        obj_mask = np.logical_and(seg==(i+1), valid)
        if np.sum(obj_mask) <= 0:
            raise Exception
        ok, obj_xyz, obj_rgb, _ = get_pts(xyz, rgb, obj_mask, num_pts=num_pts)
        obj_xyzs.append((obj_xyz + np.array([0.5, 0.5, 0.])).to(torch.float32))
        #obj_xyz = (obj_xyz - np.array([-0.5, 0, 0.58])).to(torch.float32) # [-0.475, 0, 0.58], [-0.475, 0, 0.42]
        if not ok:
            raise Exception

        if i in target_objs:
            if ignore_rgb:
                obj_pcs.append(obj_xyz)
            else:
                obj_pcs.append(torch.concat([obj_xyz, obj_rgb[:, :3]/255], dim=-1))
            obj_pad_mask.append(0)
            pc_pose = np.eye(4)
            pc_pose[:3, 3] = torch.mean(obj_xyz, dim=0).numpy()
            current_pc_poses.append(pc_pose)
        elif i in other_objs:
            if ignore_rgb:
                other_obj_pcs.append(obj_xyz)
            else:
                other_obj_pcs.append(torch.concat([obj_xyz, obj_rgb[:, :3]/255], dim=-1))
            other_obj_pad_mask.append(0)
        else:
            raise Exception

    ###################################
    # computes goal positions for objects
    # Important: because of the noises we added to point clouds, the rearranged point clouds will not be perfect
    if use_virtual_structure_frame:
        goal_structure_pose = tra.euler_matrix(structure_parameters["rotation"][0], structure_parameters["rotation"][1],
                                          structure_parameters["rotation"][2])
        goal_structure_pose[:3, 3] = [structure_parameters["position"][0], structure_parameters["position"][1],
                                 structure_parameters["position"][2]]
        goal_structure_pose_inv = np.linalg.inv(goal_structure_pose)

    goal_obj_poses = []
    current_obj_poses = []
    goal_pc_poses = []
    for obj, current_pc_pose in zip(target_objs, current_pc_poses):
        goal_pose = np.eye(4)
        current_pose = np.eye(4)
        # goal_pose = h5[obj][0]
        # current_pose = h5[obj][step_t]
        if inference_mode:
            goal_obj_poses.append(goal_pose)
            current_obj_poses.append(current_pose)

        goal_pc_pose = goal_pose @ np.linalg.inv(current_pose) @ current_pc_pose
        if use_virtual_structure_frame:
            goal_pc_pose = goal_structure_pose_inv @ goal_pc_pose
        goal_pc_poses.append(goal_pc_pose)

    # pad data
    for i in range(max_num_objects - len(target_objs)):
        obj_pcs.append(torch.zeros_like(obj_pcs[0], dtype=torch.float32))
        obj_pad_mask.append(1)
    for i in range(max_num_other_objects - len(other_objs)):
        other_obj_pcs.append(torch.zeros_like(obj_pcs[0], dtype=torch.float32))
        other_obj_pad_mask.append(1)

    ###################################
    # preparing sentence
    sentence = []
    sentence_pad_mask = []

    # structure parameters
    # 5 parameters
    sentence.append((structure_parameters["type"], "shape"))
    sentence.append((structure_parameters["rotation"][2], "rotation"))
    sentence.append((structure_parameters["position"][0], "position_x"))
    sentence.append((structure_parameters["position"][1], "position_y"))
    for _ in range(4):
        sentence_pad_mask.append(0)
    sentence.append(("PAD", None))
    sentence_pad_mask.append(1)

    ###################################
    # paddings
    for i in range(max_num_objects - len(target_objs)):
        goal_pc_poses.append(np.eye(4))

    # shuffle the position of objects
    if shuffle_object_index:
        shuffle_target_object_indices = list(range(len(target_objs)))
        random.shuffle(shuffle_target_object_indices)
        shuffle_object_indices = shuffle_target_object_indices + list(range(len(target_objs), max_num_objects))
        obj_pcs = [obj_pcs[i] for i in shuffle_object_indices]
        goal_pc_poses = [goal_pc_poses[i] for i in shuffle_object_indices]
        if inference_mode:
            goal_obj_poses = [goal_obj_poses[i] for i in shuffle_object_indices]
            current_obj_poses = [current_obj_poses[i] for i in shuffle_object_indices]
            target_objs = [target_objs[i] for i in shuffle_target_object_indices]
            current_pc_poses = [current_pc_poses[i] for i in shuffle_object_indices]

    ###################################
    if use_virtual_structure_frame:
        if ignore_distractor_objects:
            # language, structure virtual frame, target objects
            pcs = obj_pcs
            type_index = [0] * max_num_shape_parameters + [2] + [3] * max_num_objects
            position_index = list(range(max_num_shape_parameters)) + [0] + list(range(max_num_objects))
            pad_mask = sentence_pad_mask + [0] + obj_pad_mask
        else:
            # language, distractor objects, structure virtual frame, target objects
            pcs = other_obj_pcs + obj_pcs
            type_index = [0] * max_num_shape_parameters + [1] * max_num_other_objects + [2] + [3] * max_num_objects
            position_index = list(range(max_num_shape_parameters)) + list(range(max_num_other_objects)) + [0] + list(range(max_num_objects))
            pad_mask = sentence_pad_mask + other_obj_pad_mask + [0] + obj_pad_mask
        goal_poses = [goal_structure_pose] + goal_pc_poses
    else:
        if ignore_distractor_objects:
            # language, target objects
            pcs = obj_pcs
            type_index = [0] * max_num_shape_parameters + [3] * max_num_objects
            position_index = list(range(max_num_shape_parameters)) + list(range(max_num_objects))
            pad_mask = sentence_pad_mask + obj_pad_mask
        else:
            # language, distractor objects, target objects
            pcs = other_obj_pcs + obj_pcs
            type_index = [0] * max_num_shape_parameters + [1] * max_num_other_objects + [3] * max_num_objects
            position_index = list(range(max_num_shape_parameters)) + list(range(max_num_other_objects)) + list(range(max_num_objects))
            pad_mask = sentence_pad_mask + other_obj_pad_mask + obj_pad_mask
        goal_poses = goal_pc_poses

    datum = {
        "pcs": pcs,
        "sentence": sentence,
        "goal_poses": goal_poses,
        "type_index": type_index,
        "position_index": position_index,
        "pad_mask": pad_mask,
        "t": step_t,
        "filename": "", #filename
    }

    if shuffle_object_index:
        datum["shuffle_indices"] = shuffle_object_indices[:num_rearrange_objs]

    if inference_mode:
        datum["rgb"] = rgb
        datum["goal_obj_poses"] = goal_obj_poses
        datum["current_obj_poses"] = current_obj_poses
        datum["target_objs"] = target_objs
        datum["initial_scene"] = initial_scene
        #datum["ids"] = ids
        # datum["goal_specification"] = goal_specification
        datum["current_pc_poses"] = current_pc_poses

    return datum
