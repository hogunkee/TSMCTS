from utils_ur5 import *
from utils_sim2real import inverse_projection

class RealEnvironment:
    def __init__(self, args):
        self.RS = RealSense()
        self.CGN = ContactGraspNet(K=self.RS.K_rs)
        self.UR5 = UR5Robot(self.RS)
        self.GSAM = GroundedSAM()

        self.INIT_JOINTS = None
        self.current_classes = None
        self.current_obs = None

        RS_BIAS = [0, -0.01, 0] #[0.02, -0.01, -0.05]
        self.UR5.set_rs_bias(RS_BIAS)

    def reset(self, classes, move_ur5=True):
        rospy.sleep(1.0)
        if move_ur5:
            rgb, depth = self.UR5.get_view(self.UR5.ROBOT_INIT_POS, show_img=True)
        else:
            rgb, depth = self.RS.get_frames()
        detections = self.GSAM.get_masks(rgb, classes)
        segmap = np.zeros(depth.shape)
        for i, m in enumerate(detections.mask):
            segmap[m] = i+1

        rgb, depth, segmap
        rgb_resized = cv2.resize(rgb, (320, 240))
        depth_resized = cv2.resize(depth, (320, 240))
        segmap_resized = cv2.resize(segmap, (320, 240), interpolation=cv2.INTER_NEAREST)

        rgb_pad = np.pad(rgb_resized, [[60, 60], [80, 80], [0,0]])
        depth_pad = np.pad(depth_resized, [[60, 60], [80, 80]])
        segmap_pad = np.pad(segmap_resized, [[60, 60], [80, 80]])

        obs = {
                'rgb_raw': rgb,
                'depth_raw': depth,
                'segmentation_raw': segmap,
                'rgb': rgb_pad,
                'depth': depth_pad,
                'segmentation': segmap_pad
                }
        self.INIT_JOINTS = self.UR5.get_joint_states()
        self.current_classses = classes
        self.current_obs = obs
        return obs

    def check_go(self):
        x = input('go?')
        print(x)
        if x!='y' and x!='go':
            print('exit.')
            exit()

    def step(self, target_object, target_position, rot_angle):
        rgb = self.current_obs['rgb_raw']
        depth = self.current_obs['depth_raw']
        segmap = self.current_obs['segmentation_raw']

        # 1. Pick up the target object.
        grasps, scores = self.CGN.get_grasps(rgb, depth, segmap, target_object, num_K=1, show_result=True)
        print(grasps)
        grasp = grasps[0]
        #grasp_4dof = project_grasp_4dof(grasp)
        self.pick(grasp)

        # 2. Place down at the target position with rotation.
        self.place(grasp, target_position, rot_angle)

        return self.reset(self.current_classes)


    def pick(self, grasp): #, grasp_4dof
        pick_pos, pick_quat = self.UR5.get_goal_from_grasp(grasp)
        #pick_4dof_pos, pick_4dof_quat = self.UR5.get_goal_from_grasp(grasp_4dof)
        #UR5.get_view(pick_pos + np.array([0, 0, 0.1]), pick_4dof_quat)
        self.check_go()
        self.UR5.get_view(pick_pos + np.array([0, 0, 0.1]), pick_quat)
        self.check_go()
        self.UR5.get_view(pick_pos + np.array([0, 0, 0.05]), pick_quat)
        self.check_go()
        self.UR5.get_view(grasp=1.0)
        self.check_go()
        self.UR5.get_view(pick_pos + np.array([0, 0, 0.1]), pick_quat, grasp=1.0)
        self.check_go()
        #UR5.get_view(pick_pos + np.array([0, 0, 0.1]), pick_4dof_quat, grasp=1.0)
        #check_go()
        self.UR5.get_view(pick_pos + np.array([0, 0, 0.2]), grasp=1.0)
        self.check_go()
        self.UR5.move_to_joints(self.INIT_JOINTS)
        self.check_go()
        self.UR5.get_view(self.UR5.ROBOT_INIT_POS, grasp=0.0)

    def place(self, grasp, target_position, rot_angle):
        target_pose = inverse_projection(depth, target_position, self.RS.K_rs, self.RS.D_rs)
        roll, pitch, yaw = mat2euler(grasp[:3, :3])
        yaw += rot_angle
        yaw %= 2*np.pi
        quat = euler2quat([roll, pitch, yaw])
        placement = form_T(quat2mat(quat), target_pose)
        print(placement)

        place_pos, place_quat = self.UR5.get_goal_from_grasp(placement)
        self.check_go()
        self.UR5.get_view(place_pos + np.array([0, 0, 0.1]), place_quat, grasp=1.0)
        self.check_go()
        self.UR5.get_view(place_pos + np.array([0, 0, 0.05]), place_quat, grasp=1.0)
        self.check_go()
        self.UR5.get_view(grasp=0.0)
        self.check_go()
        self.UR5.get_view(place_pos + np.array([0, 0, 0.1]), place_quat, grasp=0.0)
        self.check_go()
        self.UR5.get_view(place_pos + np.array([0, 0, 0.2]), place_quat, grasp=0.0)
        self.check_go()
        self.UR5.move_to_joints(self.INIT_JOINTS)
        self.check_go()
        self.UR5.get_view(self.UR5.ROBOT_INIT_POS, grasp=0.0)


if __name__=='__main__':
    env = RealEnvironment(None)
    classes = ['Apple. Orange. Fruit.']

    obs = env.reset(classes, move_ur5=True)
    plt.imshow(obs['rgb'])#.astype(np.uint8))
    plt.show()
    #plt.imshow(obs['depth'])
    #plt.show()
    #plt.imshow(obs['segmentation'])
    #plt.show()

    x = input('where to place?')
    place = [int(p) for p in x.replace(' ', '').split(',')]
    env.step(1, np.array(place), None)
