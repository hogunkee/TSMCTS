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
        #obj_xyz = (obj_xyz - np.array([-0.5, 0, 0.58])).to(torch.float32) # [-0.475, 0, 0.58], [-0.475, 0, 0.42]
        obj_xyzs.append((obj_xyz + np.array([0.5, 0.5, 0.])).to(torch.float32))
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

