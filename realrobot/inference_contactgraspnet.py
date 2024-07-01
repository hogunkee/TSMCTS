import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2
import pyrealsense2 as rs
import time
from matplotlib import pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(BASE_DIR))
sys.path.append('/home/ur-plusle/Desktop/contact_graspnet/contact_graspnet')
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from PIL import Image
from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

def get_realsense_pipeline():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    cfg = pipeline.start(config)
    align = rs.align(rs.stream.color)
    return cfg, pipeline, align

def get_frames(pipeline, align):
    #spat_filter = rs.spatial_filter(0.40, 40.0, 4.0, 1.0)
    #temp_filter = rs.temporal_filter(0.4, 80.0, 6)

    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None

    #depth_frame = spat_filter.process(depth_frame)
    #depth_frame = temp_filter.process(depth_frame)

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data()) * 0.001
    color_image = np.asanyarray(color_frame.get_data())
    return color_image, depth_image

def inference(global_config, checkpoint_dir, img_idx, K=None, local_regions=True, skip_border_objects=False, filter_grasps=True, segmap_id=None, z_range=[0.2,1.8], forward_passes=1):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    
    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')
    
    os.makedirs('results', exist_ok=True)

    # Get RGB-D images from RealSense camera
    rs_cfg, rs_pipeline, rs_align = get_realsense_pipeline()
    time.sleep(2)
    rgb, depth = get_frames(rs_pipeline, rs_align)
    #rgb, depth = get_frames(rs_pipeline, rs_align)
    plt.imshow(rgb)
    plt.show()
    plt.imshow(depth, vmax=1.0)
    plt.show()
    
    # Get camera instrinsics
    color_profile = rs_cfg.get_stream(rs.stream.color, 0)
    intr = color_profile.as_video_stream_profile().get_intrinsics()
    #depth_profile = rs_cfg.get_stream(rs.stream.depth, 0)
    #depth_intr = profile.as_video_stream_profile().get_intrinsics()
    fx, fy, height, width = intr.fx, intr.fy, intr.height, intr.width
    cx, cy = intr.ppx, intr.ppy
    cam_K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
    segmap = None

    if False:
        # Process example test scenes
        segmap = None
        image_dir = '/home/gun/Desktop/ur5_manipulation/object_wise/dqn/test_scenes/'
        rgb = np.array(Image.open(os.path.join(image_dir, 'goal', '%d.png' %img_idx)))
        depth = np.load(os.path.join(image_dir, 'goal', '%d.npy' %img_idx))

        fovy = 45
        img_height = 480
        f = 0.5 * img_height / np.tan(fovy * np.pi / 360)
        cam_K = np.array([[f, 0, 239.5],
                          [0, f, 239.5],
                          [0, 0, 1]])
    print(cam_K)

    pc_segments = {}
    pc_full = None
    pc_colors = None
    #segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)

    if segmap is None and (local_regions or filter_grasps):
        raise ValueError('Need segmentation map to extract local regions or filter grasps')

    if pc_full is None:
        print('Converting depth to point cloud(s)...')
        pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                            skip_border_objects=skip_border_objects, z_range=z_range)

    #visualize_grasps(pc_full, [], [], plot_opencv_cam=True, pc_colors=pc_colors)
    #input("continue?")

    print('Generating Grasps...')
    pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full,
                                    pc_segments=pc_segments, local_regions=local_regions,
                                    filter_grasps=filter_grasps, forward_passes=forward_passes)

    print(pred_grasps_cam)
    print(contact_pts)
    # Save results
    np.savez('results/predictions_{}'.format(img_idx),
              pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)

    # Visualize results
    show_image(rgb, segmap)
    visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
        
    if not glob.glob(input_paths):
        print('No files found: ', input_paths)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--img_idx', default=100, type=int)
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.0], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))

    inference(global_config, FLAGS.ckpt_dir, FLAGS.img_idx, z_range=eval(str(FLAGS.z_range)),
                K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps, segmap_id=FLAGS.segmap_id, 
                forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects)

