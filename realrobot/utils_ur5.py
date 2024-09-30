import rospy
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering
from utils_contactgraspnet import RealSense, ContactGraspNet, form_T
from utils_gsam import GroundedSAM

# moveit planner
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import String

# robotiq gripper
import pyRobotiqGripper

# ik solver
sys.path.append('/home/ur-plusle/Desktop/ikfastpy')
import ikfastpy

from transform_utils import mat2pose, quat2mat, mat2euler, euler2quat, mat2quat

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

        if True:
            theta = np.pi/16
            self.PRE_GRASP_POS_1 = np.array([0.15, -0.3, 0.55])
            self.PRE_GRASP_POS_2 = np.array([0., -0.35, 0.55])
            self.PRE_PLACE_POS = np.array([0., -0.3, 0.6])
            self.ROBOT_INIT_POS = np.array([0.0, -0.35, 0.72])
        elif False:
            theta = np.pi/8
            self.PRE_GRASP_POS_1 = np.array([0.1, -0.3, 0.55])
            #self.PRE_GRASP_POS_1 = np.array([0.2, -0.3, 0.55])
            self.PRE_GRASP_POS_2 = np.array([0., -0.35, 0.55])
            self.PRE_PLACE_POS = np.array([0., -0.3, 0.6])
            #self.PRE_PLACE_POS = np.array([0.2, -0.3, 0.6])
            self.ROBOT_INIT_POS = np.array([0.0, -0.2, 0.7])
            #self.ROBOT_INIT_POS = np.array([0.0, -0.3, 0.6])
            #self.ROBOT_INIT_POS = np.array([0.0, -0.4, 0.6])
        self.ROBOT_INIT_ROTATION = np.array([[1., 0., 0.], 
            [0., -np.cos(theta), -np.sin(theta)], 
            [0., np.sin(theta), -np.cos(theta)]])
        self.ROBOT_INIT_QUAT = mat2quat(self.ROBOT_INIT_ROTATION)
        #print("INIT Quat:")
        #print(mat2quat(self.ROBOT_INIT_ROTATION))

        #self.ROBOT_INIT_POS = np.array([0.0, -0.4, 0.6])
        #self.ROBOT_INIT_ROTATION = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])

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
        #self.T_eef_to_rs[2, 3] -= 0.10
        print('T_eef_to_rs:', self.T_eef_to_rs)
        
        # IK solver
        self.ur5_kin = ikfastpy.PyKinematics()
        self.n_joints = 6

    def solve_ik(self, ee_pose, num_solve=1):
        # ee_pose: 3x4 rigid transform matrix
        for n in range(num_solve):
            if ee_pose.shape[0]==4 and ee_pose.shape[1]==4:
                ee_pose = ee_pose[:3]
            if n>0:
                ee_pose[-1][-1] += 0.001
            ee_pose = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).dot(ee_pose)
            #print(ee_pose)
            joint_configs = self.ur5_kin.inverse(ee_pose.reshape(-1).tolist())
            n_solutions = int(len(joint_configs) / self.n_joints)
            joint_configs = np.asarray(joint_configs).reshape(n_solutions, self.n_joints)

            # find the cloest solution
            current_joint = self.get_joint_states()
            #print('current:', current_joint)
            current_yaw = current_joint[-1]
            diffs = []
            joint_candidates = []
            for joint in joint_configs:
                # joint constraints
                if not (joint[0] < 0 and joint[0] > -np.pi):
                    continue
                if not (joint[2] > 0):
                    continue
                if not (joint[3] > -1.5 * np.pi and joint[3] < 0.2):
                    continue
                if not (joint[4] > -np.pi and joint[4] < 0):
                    continue

                #print(joint)
                yaw = joint[-1]
                yaw_candidates = np.array([yaw - 2*np.pi, yaw, yaw + 2*np.pi])
                yaw_nearest = yaw_candidates[np.argmin((current_yaw - yaw_candidates)**2)]
                joint[-1] = yaw_nearest
                priority = np.array([1, 1, 1, 1, 10, 1])
                diff = np.linalg.norm(priority * (current_joint - joint))
                diffs.append(diff)
                joint_candidates.append(joint)
            if len(diffs)==0:
                return []
            idx = diffs.index(min(diffs))
            final_joint_config = joint_candidates[idx]
            #final_joint_config = joint_configs[idx]
            print("Find solutions:", final_joint_config)
            return final_joint_config

    def set_rs_bias(self, bias):
        print("Add bias:", bias)
        self.T_eef_to_rs[:3, 3] += bias

    def get_joint_states(self):
        return self.move_group.get_current_joint_values()

    def get_eef_pose(self):
        pose = self.move_group.get_current_pose().pose
        position = [pose.position.x, pose.position.y, pose.position.z]
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        return position, quaternion

    def move_to_joints(self, joints):
        self.move_group.set_joint_value_target(joints)
        res = self.move_group.plan()
        # check success
        if res[0] is False:
            print("Failed planning to the goal")
            return None, None
        else:
            # move to the view
            self.move_group.execute(res[1], wait=True)


    def get_view(self, goal_pos=None, quat=[1, 0, 0, 0], grasp=0.0, show_img=False, num_solve=1):
        # quat: xyzw
        if goal_pos is not None:
            is_feasible = False
            while not is_feasible:
                goal_pos[2] = np.clip(goal_pos[2], 0.206, 0.7) #0.21
                goal_P = form_T(quat2mat(quat), goal_pos)
                joints = self.solve_ik(goal_P, num_solve)
                if len(joints)==0:
                    pose_goal = geometry_msgs.msg.Pose()
                    pose_goal.orientation.x = quat[0]
                    pose_goal.orientation.y = quat[1]
                    pose_goal.orientation.z = quat[2]
                    pose_goal.orientation.w = quat[3]
                    pose_goal.position.x = goal_pos[0]
                    pose_goal.position.y = goal_pos[1]
                    pose_goal.position.z = goal_pos[2]
                    self.move_group.set_pose_target(pose_goal)
                else:
                    print(goal_pos)
                    print(quat)
                    self.move_group.set_joint_value_target(joints)
                res = self.move_group.plan()
                traj_length = len(res[1].joint_trajectory.points)
                print("Trajectory:", traj_length)
                # check success
                if res[0] is False:
                    print("Failed planning to the goal.")
                elif traj_length > 10:
                    print("A wrong path is obtained.")
                else:
                    print("Find a feasible trajectory.")
                    is_feasible = True
            # move to the view
            self.move_group.execute(res[1], wait=True)

        # gripper control
        if grasp > 0.0:
            self.gripper_controller.close(force=100)
            # suction gripper grasping
            # gripper_controller.grasp()
        else:
            self.gripper_controller.open(force=100)
        
        rospy.sleep(0.5)
        if show_img:
            color, depth = self.rs.get_frames()
            # plt.imshow(color)
            # plt.show()
            return color, depth
        return None, None

    def get_plan(self, goal_pos=None, quat=[1, 0, 0, 0], grasp=0.0, show_img=False, num_solve=1):
        # quat: xyzw
        if goal_pos is not None:
            goal_pos[2] = np.clip(goal_pos[2], 0.206, 0.7) #0.21
            goal_P = form_T(quat2mat(quat), goal_pos)
            joints = self.solve_ik(goal_P, num_solve)
            if len(joints)==0:
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
            else:
                print(goal_pos)
                print(quat)
                self.move_group.set_joint_value_target(joints)
                res = self.move_group.plan()
            return res
    
    def get_goal_from_grasp(self, grasp, eef_P=None):
        T_eef_to_grasp = np.dot(self.T_eef_to_rs, grasp)

        if eef_P is None:
            eef_pose, eef_quat = self.get_eef_pose()
        else:
            eef_pose, eef_quat = eef_P
        T_robot_to_eef = form_T(quat2mat(eef_quat), eef_pose)
        T_robot_to_grasp = np.dot(T_robot_to_eef, T_eef_to_grasp)
        print('goal P:', T_robot_to_grasp)

        pos, quat = mat2pose(T_robot_to_grasp)
        return pos, quat
        #print('pose')
        #print(pos)
        #print('quat')
        #print(quat)
        #return self.get_view(pos, quat, show_img=True)

    def get_T_robot_to_rs(self):
        eef_pose, eef_quat = self.get_eef_pose()
        T_robot_to_eef = form_T(quat2mat(eef_quat), eef_pose)
        T_robot_to_rs = np.dot(T_robot_to_eef, self.T_eef_to_rs)
        return T_robot_to_rs


def project_grasp_4dof(grasp):
    grasp = grasp.copy()
    roll, pitch, yaw = mat2euler(grasp[:3, :3])
    x,y,z,w = euler2quat([0, 0, yaw])
    rot_4dof = quat2mat([x,y,z,w])
    grasp[:3, :3] = rot_4dof
    return grasp

def check_go():
    x = input('go?')
    print(x)
    if x!='y' and x!='go':
        print('exit.')
        exit()

if __name__=='__main__':
    n_obj = 3
    z_thres = 0.49 #0.54 ##0.34

    RS = RealSense()
    CGN = ContactGraspNet(K=RS.K_rs)
    UR5 = UR5Robot(RS)
    GSAM = GroundedSAM()

    rospy.sleep(1.0)
    rgb, depth = UR5.get_view(UR5.ROBOT_INIT_POS, UR5.ROBOT_INIT_QUAT, show_img=True)
    #rgb, depth = RS.get_frames()
    INIT_JOINTS = UR5.get_joint_states()
    INIT_EEF_P = UR5.get_eef_pose()

    classes = ["Orange", "Apple", "Lemon"]
    detections = GSAM.get_masks(rgb, classes)
    #print(detections.class_id)

    segmap = np.zeros(depth.shape)
    for i, m in enumerate(detections.mask):
        segmap[m] = i+1
    plt.imshow(segmap)
    plt.show()

    ## Spectral Clustering ##
    # segmap, mask = CGN.get_masks(rgb, depth, n_cluster=n_obj, thres=z_thres)

    #RS_BIAS = [0.02, -0.01, -0.05]
    #UR5.set_rs_bias(RS_BIAS)

    for segmap_id in range(1, n_obj+1):
        grasps, scores = CGN.get_grasps(rgb, depth, segmap, segmap_id, num_K=1, show_result=True)
        #grasps, scores = CGN.get_4dof_grasps(rgb, depth, segmap, segmap_id, num_K=1, show_result=True)
        print(grasps)
        grasp = grasps[0]
        grasp_4dof = project_grasp_4dof(grasp)

        check_go()
        UR5.get_view(UR5.PRE_PLACE_POS, [1,0,0,0])
        check_go()
        pick_pos, pick_quat = UR5.get_goal_from_grasp(grasp, INIT_EEF_P)
        pick_4dof_pos, pick_4dof_quat = UR5.get_goal_from_grasp(grasp_4dof, INIT_EEF_P)
        #UR5.get_view(pick_pos + np.array([0, 0, 0.1]), pick_4dof_quat)
        #check_go()
        UR5.get_view(pick_pos + np.array([0, 0, 0.1]), pick_quat)
        check_go()
        UR5.get_view(pick_pos + np.array([0, 0, 0.05]), pick_quat)
        check_go()
        UR5.get_view(grasp=1.0)
        check_go()
        UR5.get_view(pick_pos + np.array([0, 0, 0.1]), pick_quat, grasp=1.0)
        check_go()
        #UR5.get_view(pick_pos + np.array([0, 0, 0.1]), pick_4dof_quat, grasp=1.0)
        #check_go()
        UR5.get_view(pick_pos + np.array([0, 0, 0.2]), UR5.ROBOT_INIT_QUAT, grasp=1.0)
        check_go()
        UR5.move_to_joints(INIT_JOINTS)
        check_go()
        UR5.get_view(UR5.ROBOT_INIT_POS, UR5.ROBOT_INIT_QUAT, grasp=0.0)
        break

