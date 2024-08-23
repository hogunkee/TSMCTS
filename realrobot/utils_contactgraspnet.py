# RealSense Setup #
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from utils_sim2real import *
from Pose_Estimation_Class import *
from transform_utils import mat2euler, quat2mat


class RealSense:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        # Start streaming
        self.cfg = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        # Get camera instrinsics
        color_profile = self.cfg.get_stream(rs.stream.color, 0)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        fx, fy, height, width = intr.fx, intr.fy, intr.height, intr.width
        cx, cy = intr.ppx, intr.ppy
        self.K_rs = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
        self.D_rs = 0
        rospy.sleep(2.0)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()) * 0.001
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

import sys
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True) 
sys.path.append('/home/ur-plusle/Desktop/contact_graspnet/contact_graspnet')
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from PIL import Image
from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

class ContactGraspNet:
    def __init__(self, ckpt_dir = '/home/ur-plusle/Desktop/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001', K=None):
        global_config = config_utils.load_config(ckpt_dir, batch_size=1, arg_configs=[])
        self.K_rs = K

        # Build the model
        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        # Load weights
        self.grasp_estimator.load_weights(self.sess, saver, ckpt_dir, mode='test')

    def get_grasps(self, rgb, depth, segmap=None, segmap_id=-1, num_K=10, show_result=True):
        # os.makedirs('results', exist_ok=True)
        local_regions = False
        filter_grasps = True #False
        skip_border_objects = True
        forward_passes = 1
        z_range = [0.2, 1.0]

        pc_segments = {}
        pc_full = None
        pc_colors = None

        if segmap is None and (local_regions or filter_grasps):
            raise ValueError('Need segmentation map to extract local regions or filter grasps')

        if pc_full is None:
            print('Converting depth to point cloud(s)...')
            pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(
                                depth, self.K_rs, segmap=segmap, rgb=rgb, segmap_id=segmap_id, 
                                skip_border_objects=skip_border_objects, z_range=z_range)

        print('Generating Grasps...')
        pred_grasps_cam, pred_scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(
                                        self.sess, pc_full, pc_segments=pc_segments, 
                                        local_regions=local_regions, filter_grasps=filter_grasps,
                                        forward_passes=forward_passes)
        if segmap_id not in pred_grasps_cam:
            print('# grasps: 0')
            return [], []
        grasps = pred_grasps_cam[segmap_id]
        scores = pred_scores[segmap_id]
        print('# grasps:', len(grasps))

        def get_theta(grasp):
            cos_theta = (grasp[:3, :3].dot(np.array([[0, 0, 1]]).T).T[0]).dot(np.array([0, 0, 1]))
            return cos_theta
        grasp_over07 = [(g, s) for g, s in zip(grasps, scores) if get_theta(g) > 0.7]
        if len(grasp_over07)==0:
            grasp_over03 = [(g, s) for g, s in zip(grasps, scores) if get_theta(g) > 0.3]
            if len(grasp_over03)==0:
                grasp_over0 = [(g, s) for g, s in zip(grasps, scores) if get_theta(g) > 0.0]
                if len(grasp_over0)==0:
                    return [], []
                else:
                    grasps, scores = zip(*grasp_over0)
            else:
                grasps, scores = zip(*grasp_over03)
        else:
            grasps, scores = zip(*grasp_over07)
        #grasps, scores = zip(*[(g, s) for g, s in zip(grasps, scores) if get_theta(g) > 0.7]) #0.8
        #grasps, scores = zip(*[(g, s) for g, s in zip(grasps, scores) if (np.trace(g[:3, :3])-1) / 2 > 0.8])

        grasps, scores = zip(*sorted(zip(grasps, scores), key=lambda x: x[1]))
        filtered_grasps_cam = {segmap_id: grasps[:num_K]}
        filtered_scores = {segmap_id: scores[:num_K]}

        if show_result:
            # Visualize results
            show_image(rgb, segmap)
            #visualize_grasps(pc_full, pred_grasps_cam, pred_scores, plot_opencv_cam=True, pc_colors=pc_colors)
            visualize_grasps(pc_full, filtered_grasps_cam, filtered_scores, plot_opencv_cam=True, pc_colors=pc_colors)
        return filtered_grasps_cam[segmap_id], filtered_scores[segmap_id]

    def get_4dof_grasps(self, rgb, depth, segmap=None, segmap_id=-1, num_K=10, show_result=True):
        # os.makedirs('results', exist_ok=True)
        local_regions = False
        filter_grasps = True #False
        skip_border_objects = True
        forward_passes = 1
        z_range = [0.2, 1.0]

        pc_segments = {}
        pc_full = None
        pc_colors = None

        if segmap is None and (local_regions or filter_grasps):
            raise ValueError('Need segmentation map to extract local regions or filter grasps')

        if pc_full is None:
            print('Converting depth to point cloud(s)...')
            pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(
                                depth, self.K_rs, segmap=segmap, rgb=rgb, segmap_id=segmap_id, 
                                skip_border_objects=skip_border_objects, z_range=z_range)

        print('Generating Grasps...')
        pred_grasps_cam, pred_scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(
                                        self.sess, pc_full, pc_segments=pc_segments, 
                                        local_regions=local_regions, filter_grasps=filter_grasps,
                                        forward_passes=forward_passes)
        grasps = pred_grasps_cam[segmap_id]
        scores = pred_scores[segmap_id]
        print('# grasps:', len(grasps))

        def get_theta(grasp):
            cos_theta = (grasp[:3, :3].dot(np.array([[0, 0, 1]]).T).T[0]).dot(np.array([0, 0, 1]))
            return cos_theta
        grasps, scores = zip(*[(g, s) for g, s in zip(grasps, scores) if get_theta(g) > 0.8])
        #grasps, scores = zip(*[(g, s) for g, s in zip(grasps, scores) if (np.trace(g[:3, :3])-1) / 2 > 0.8])
        grasps, scores = zip(*sorted(zip(grasps, scores), key=lambda x: x[1]))
        for g in grasps:
            roll, pitch, yaw = mat2euler(g[:3, :3])
            x,y,z,w = euler2quat([0, 0, yaw])
            rot_4dof = quat2mat([x,y,z,w])
            g[:3, :3] = rot_4dof
        filtered_grasps_cam = {segmap_id: grasps[:num_K]}
        filtered_scores = {segmap_id: scores[:num_K]}

        if show_result:
            # Visualize results
            show_image(rgb, segmap)
            #visualize_grasps(pc_full, pred_grasps_cam, pred_scores, plot_opencv_cam=True, pc_colors=pc_colors)
            visualize_grasps(pc_full, filtered_grasps_cam, filtered_scores, plot_opencv_cam=True, pc_colors=pc_colors)
        return filtered_grasps_cam[segmap_id], filtered_scores[segmap_id]

    def get_masks(self, color, depth, n_cluster=3, thres=0.625):
        fmask = (depth < thres).astype(float)
        H, W = fmask.shape
        fmask = cv2.resize(fmask, (64, 48), interpolation=cv2.INTER_NEAREST)
        #plt.imshow(fmask)
        #plt.show()

        my, mx = np.nonzero(fmask)
        points = list(zip(mx, my, np.ones_like(mx) * W))
        z = (np.array(points).T / np.linalg.norm(points, axis=1)).T

        im_blur = cv2.blur(color, (5, 5))
        colors = np.array([im_blur[y, x] / (1*255) for x, y in zip(mx, my)])
        z_color = np.concatenate([z, colors], 1)
        clusters = SpectralClustering(n_clusters=n_cluster, n_init=10).fit_predict(z_color)

        new_mask = np.zeros([fmask.shape[0], fmask.shape[1]])#, n_cluster])
        for x, y, c in zip(mx, my, clusters):
            new_mask[y, x] = c+1
            #new_mask[y, x, c] = 1
        masks = new_mask.astype(float)
        #masks = new_mask.transpose([2,0,1]).astype(float)

        masks = cv2.resize(masks, (W, H), interpolation=cv2.INTER_NEAREST)
        fmask = cv2.resize(fmask, (W, H), interpolation=cv2.INTER_NEAREST)

        return masks, fmask


if __name__=='__main__':
    RS = RealSense()
    CGN = ContactGraspNet(K=RS.K_rs)
    rospy.sleep(1.0)
    n_obj = 3
    z_thres = 0.54 #0.34

    rgb, depth = RS.get_frames()
    segmap, mask = CGN.get_masks(rgb, depth, n_cluster=n_obj, thres=z_thres)
    plt.imshow(segmap)
    plt.show()

    for segmap_id in range(1, n_obj+1):
        grasps, scores = CGN.get_grasps(rgb, depth, segmap, segmap_id, num_K=10, show_result=True)

if False:
    RS = RealSense()
    CGN = ContactGraspNet(K=RS.K_rs)
    rospy.sleep(1.0)

    rgb, depth = RS.get_frames()
    segmap = (depth<0.34).astype(int) #None
    #plt.imshow(segmap)
    #plt.show()
    segmap_id = 1
    CGN.get_grasps(rgb, depth, segmap, segmap_id, show_result=True)
