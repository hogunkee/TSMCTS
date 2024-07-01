import rospy
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering
from utils_contactgraspnet import RealSense, ContactGraspNet, form_T

# moveit planner
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import String

# robotiq gripper
import pyRobotiqGripper

from transform_utils import mat2pose, quat2mat #mat2quat

try:
    from math import pi, tau, dist, fabs, cos
except:
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

class UR5Robot:
    def __init__(self, realsense):
        self.rs = realsense

        #ARM_JOINT_NAME = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.ROBOT_INIT_POS = [0.0, -0.3, 0.65]
        self.ROBOT_INIT_ROTATION = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])

        # init node
        rospy.init_node("test", anonymous=True)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("manipulator")

        # init gripper
        self.gripper_controller = pyRobotiqGripper.RobotiqGripper()
        self.gripper_controller.activate()

        # eef to realsense offset
        self.T_eef_to_rs = np.load('rs_extrinsic.npy')
        self.T_eef_to_rs[2, 3] += 0.15
        print('T_eef_to_rs:', self.T_eef_to_rs)

    def get_joint_states(self):
        return self.move_group.get_current_joint_values()

    def get_eef_pose(self):
        pose = self.move_group.get_current_pose().pose
        position = [pose.position.x, pose.position.y, pose.position.z]
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        return position, quaternion

    def get_view(self, goal_pos=None, quat=[1, 0, 0, 0], grasp=0.0, show_img=False):
        # quat: xyzw
        if goal_pos is not None:
            pose_goal = geometry_msgs.msg.Pose()
            pose_goal.orientation.x = quat[0]
            pose_goal.orientation.y = quat[1]
            pose_goal.orientation.z = quat[2]
            pose_goal.orientation.w = quat[3]
            pose_goal.position.x = goal_pos[0]
            pose_goal.position.y = goal_pos[1]
            pose_goal.position.z = goal_pos[2]
            self.move_group.set_pose_target(pose_goal)
            res = self.move_group.plan()

            # check success
            if res[0] is False:
                print("Failed planning to the goal")
                return None, None
            else:
                # move to the view
                self.move_group.execute(res[1], wait=True)
        # gripper control
        if grasp > 0.0:
            self.gripper_controller.close()
            # suction gripper grasping
            # gripper_controller.grasp()
        else:
            self.gripper_controller.open()
        
        rospy.sleep(0.5)
        if show_img:
            color, depth = self.rs.get_frames()
            # plt.imshow(color)
            # plt.show()
            return color, depth
        return None, None
    
    def move_to_grasp(self, grasp):
        T_eef_to_grasp = np.dot(self.T_eef_to_rs, grasp)

        eef_pose, eef_quat = self.get_eef_pose()
        T_robot_to_eef = form_T(quat2mat(eef_quat), eef_pose)
        T_robot_to_grasp = np.dot(T_robot_to_eef, T_eef_to_grasp)
        print('goal P:', T_robot_to_grasp)

        pos, quat = mat2pose(T_robot_to_grasp)
        #print('pose')
        #print(pos)
        #print('quat')
        #print(quat)
        return self.get_view(pos, quat, show_img=True)


if __name__=='__main__':
    n_obj = 3
    z_thres = 0.54 ##0.34

    RS = RealSense()
    CGN = ContactGraspNet(K=RS.K_rs)
    UR5 = UR5Robot(RS)

    rospy.sleep(1.0)
    rgb, depth = UR5.get_view(UR5.ROBOT_INIT_POS, show_img=True)

    #rgb, depth = RS.get_frames()
    segmap, mask = CGN.get_masks(rgb, depth, n_cluster=n_obj, thres=z_thres)
    plt.imshow(segmap)
    plt.show()

    for segmap_id in range(1, n_obj+1):
        grasps, scores = CGN.get_grasps(rgb, depth, segmap, segmap_id, num_K=1, show_result=True)
        print(grasps)
        UR5.move_to_grasp(grasps[0])
        UR5.get_view(grasp=1.0)
        UR5.get_view(UR5.ROBOT_INIT_POS, grasp=1.0)
        break

