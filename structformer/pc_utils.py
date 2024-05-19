import os
import sys
import numpy as np
import random
import torch

from structformer.utils.rearrangement import get_pts

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


def get_raw_data(obs, env, structure_param, view='top', max_num_objects=10, num_pts=1024):
    # filename, t = self.arrangement_data[idx]

    # h5 = h5py.File(filename, 'r')
    # ids = self._get_ids(h5)
    # # moved_objs = h5['moved_objs'][()].split(',')
    # all_objs = sorted([o for o in ids.keys() if "object_" in o])
    # goal_specification = json.loads(str(np.array(h5["goal_specification"])))
    # num_rearrange_objs = len(goal_specification["rearrange"]["objects"])
    # # all_object_specs = goal_specification["rearrange"]["objects"] + goal_specification["anchor"]["objects"] + \
    # #                    goal_specification["distract"]["objects"]

    # ###################################
    # # getting scene images and point clouds
    # scene = self._get_images(h5, t, ee=True)
    
    #rgb, depth, seg, valid, xyz = scene
    rgb = obs[view]['rgb']
    depth = obs[view]['depth']
    seg = obs[view]['segmentation']

    ids = {'table': 1}
    for id, tobj in env.table_objects_list.items():
        ids[tobj[0]] = id
    all_objs = [tobj[0] for id, tobj in env.table_objects_list.items()]
    num_rearrange_objs = len(env.table_objects_list)

    valid = np.logical_and(seg > 0, seg < 1000)
    if view=='top':
        xyz = env.camera.rgbd_2_world_batch(env.camera.origin_depth) #depth)
    else:
        xyz = env.camera_front_top.rgbd_2_world_batch(env.camera_front_top.origin_depth) #depth)

    # getting object point clouds
    obj_xyzs = []
    obj_rgbs = []
    object_pad_mask = []
    rearrange_obj_labels = []
    for i, obj in enumerate(all_objs):
        obj_mask = np.logical_and(seg == ids[obj], valid)
        if np.sum(obj_mask) <= 0:
            raise Exception
        ok, obj_xyz, obj_rgb, _ = get_pts(xyz, rgb, obj_mask, num_pts=num_pts)
        obj_xyzs.append((obj_xyz - np.array([-0.475, 0, 0.625])).to(torch.float32))
        obj_rgbs.append(obj_rgb[:, :3]/255)
        object_pad_mask.append(0)
        if i < num_rearrange_objs:
            rearrange_obj_labels.append(1.0)
        else:
            rearrange_obj_labels.append(0.0)

    # pad data
    for i in range(max_num_objects - len(all_objs)):
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
    shuffle_object_indices = list(range(len(all_objs)))
    random.shuffle(shuffle_object_indices)
    shuffle_object_indices = shuffle_object_indices + list(range(len(all_objs), max_num_objects))
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
        "shuffle_indices": shuffle_object_indices[:len(all_objs)]
    }

    return datum


#def get_diffusion_data(self, idx, inference_mode=False, shuffle_object_index=False):
def get_diffusion_data(obs, env, structure_param, view='top', inference_mode=False, \
                        shuffle_object_index=False, max_num_objects=10, num_pts=1024):
    # filename, _ = self.arrangement_data[idx]

    # h5 = h5py.File(filename, 'r')
    # ids = self._get_ids(h5)
    # all_objs = sorted([o for o in ids.keys() if "object_" in o])
    # goal_specification = json.loads(str(np.array(h5["goal_specification"])))
    # num_rearrange_objs = len(goal_specification["rearrange"]["objects"])
    # num_other_objs = len(goal_specification["anchor"]["objects"] + goal_specification["distract"]["objects"])
    # assert len(all_objs) == num_rearrange_objs + num_other_objs, "{}, {}".format(len(all_objs), num_rearrange_objs + num_other_objs)
    # assert num_rearrange_objs <= self.max_num_objects
    # assert num_other_objs <= self.max_num_other_objects

    # important: only using the last step
    # step_t = num_rearrange_objs

    # target_objs = all_objs[:num_rearrange_objs]
    # other_objs = all_objs[num_rearrange_objs:]

    # structure_parameters = goal_specification["shape"]

    # Important: ensure the order is correct
    # if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
    #     target_objs = target_objs[::-1]
    # elif structure_parameters["type"] == "tower" or structure_parameters["type"] == "dinner":
    #     target_objs = target_objs
    # else:
    #     raise KeyError("{} structure is not recognized".format(structure_parameters["type"]))
    # all_objs = target_objs + other_objs

    ###################################
    use_virtual_structure_frame = True
    ignore_distractor_objects = True
    max_num_shape_parameters = 7
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
    scene = self._get_images(h5, step_t, ee=True)
    #rgb, depth, seg, valid, xyz = scene
    rgb = obs[view]['rgb']
    depth = obs[view]['depth']
    seg = obs[view]['segmentation']

    ids = {'table': 1}
    for id, tobj in env.table_objects_list.items():
        ids[tobj[0]] = id
    all_objs = [tobj[0] for id, tobj in env.table_objects_list.items()]
    num_rearrange_objs = len(env.table_objects_list)
    step_t = num_rearrange_objs

    valid = np.logical_and(seg > 0, seg < 1000)
    if view=='top':
        xyz = env.camera.rgbd_2_world_batch(env.camera.origin_depth) #depth)
    else:
        xyz = env.camera_front_top.rgbd_2_world_batch(env.camera_front_top.origin_depth) #depth)


    if inference_mode:
        initial_scene = scene

    # getting object point clouds
    obj_pcs = []
    obj_pad_mask = []
    current_pc_poses = []
    other_obj_pcs = []
    other_obj_pad_mask = []
    for obj in all_objs:
        obj_mask = np.logical_and(seg == ids[obj], valid)
        if np.sum(obj_mask) <= 0:
            raise Exception
        ok, obj_xyz, obj_rgb, _ = get_pts(xyz, rgb, obj_mask, num_pts=num_pts)
        if not ok:
            raise Exception

        if obj in target_objs:
            if self.ignore_rgb:
                obj_pcs.append(obj_xyz)
            else:
                obj_pcs.append(torch.concat([obj_xyz, obj_rgb], dim=-1))
            obj_pad_mask.append(0)
            pc_pose = np.eye(4)
            pc_pose[:3, 3] = torch.mean(obj_xyz, dim=0).numpy()
            current_pc_poses.append(pc_pose)
        elif obj in other_objs:
            if self.ignore_rgb:
                other_obj_pcs.append(obj_xyz)
            else:
                other_obj_pcs.append(torch.concat([obj_xyz, obj_rgb], dim=-1))
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
        goal_pose = h5[obj][0]
        current_pose = h5[obj][step_t]
        if inference_mode:
            goal_obj_poses.append(goal_pose)
            current_obj_poses.append(current_pose)

        goal_pc_pose = goal_pose @ np.linalg.inv(current_pose) @ current_pc_pose
        if use_virtual_structure_frame:
            goal_pc_pose = goal_structure_pose_inv @ goal_pc_pose
        goal_pc_poses.append(goal_pc_pose)

    # transform current object point cloud to the goal point cloud in the world frame
    if self.debug:
        new_obj_pcs = [copy.deepcopy(pc.numpy()) for pc in obj_pcs]
        for i, obj_pc in enumerate(new_obj_pcs):

            current_pc_pose = current_pc_poses[i]
            goal_pc_pose = goal_pc_poses[i]
            if self.use_virtual_structure_frame:
                goal_pc_pose = goal_structure_pose @ goal_pc_pose
            print("current pc pose", current_pc_pose)
            print("goal pc pose", goal_pc_pose)

            goal_pc_transform = goal_pc_pose @ np.linalg.inv(current_pc_pose)
            print("transform", goal_pc_transform)
            new_obj_pc = copy.deepcopy(obj_pc)
            new_obj_pc[:, :3] = trimesh.transform_points(obj_pc[:, :3], goal_pc_transform)
            print(new_obj_pc.shape)

            # visualize rearrangement sequence (new_obj_xyzs), the current object before moving (obj_xyz), and other objects
            new_obj_pcs[i] = new_obj_pc
            new_obj_pcs[i][:, 3:] = np.tile(np.array([1, 0, 0], dtype=np.float), (new_obj_pc.shape[0], 1))
            new_obj_rgb_current = np.tile(np.array([0, 1, 0], dtype=np.float), (new_obj_pc.shape[0], 1))
            show_pcs([pc[:, :3] for pc in new_obj_pcs] + [pc[:, :3] for pc in other_obj_pcs] + [obj_pc[:, :3]],
                     [pc[:, 3:] for pc in new_obj_pcs] + [pc[:, 3:] for pc in other_obj_pcs] + [new_obj_rgb_current],
                     add_coordinate_frame=True)
        show_pcs([pc[:, :3] for pc in new_obj_pcs], [pc[:, 3:] for pc in new_obj_pcs], add_coordinate_frame=True)

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

    ###################################
    if self.debug:
        print("---")
        print("all objects:", all_objs)
        print("target objects:", target_objs)
        print("other objects:", other_objs)
        print("goal specification:", goal_specification)
        print("sentence:", sentence)
        show_pcs([pc[:, :3] for pc in obj_pcs + other_obj_pcs], [pc[:, 3:] for pc in obj_pcs + other_obj_pcs], add_coordinate_frame=True)

    assert len(obj_pcs) == len(goal_pc_poses)
    ###################################

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
        "filename": filename
    }

    if shuffle_object_index:
        datum["shuffle_indices"] = shuffle_object_indices[:len(all_objs)]

    if inference_mode:
        datum["rgb"] = rgb
        datum["goal_obj_poses"] = goal_obj_poses
        datum["current_obj_poses"] = current_obj_poses
        datum["target_objs"] = target_objs
        datum["initial_scene"] = initial_scene
        datum["ids"] = ids
        datum["goal_specification"] = goal_specification
        datum["current_pc_poses"] = current_pc_poses

    return datum
