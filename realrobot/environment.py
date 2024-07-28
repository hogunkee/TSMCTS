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

        RS_BIAS = [0, 0, 0] #[0, -0.01, 0] #[0.02, -0.01, -0.05]
        self.UR5.set_rs_bias(RS_BIAS)

        self.default_depth = None
        self.init_eef_P = None

    def set_default_depth(self, move_ur5=True):
        rospy.sleep(1.0)
        if move_ur5:
            rgb, depth = self.UR5.get_view(self.UR5.ROBOT_INIT_POS, self.UR5.ROBOT_INIT_QUAT, 
                                            show_img=True)
        else:
            rgb, depth = self.RS.get_frames()
        np.save('default_depth.npy', depth)

    def get_observation(self, move_ur5=True):
        rospy.sleep(1.0)
        if move_ur5:
            rgb, depth = self.UR5.get_view(self.UR5.ROBOT_INIT_POS, self.UR5.ROBOT_INIT_QUAT,
                                            grasp=0.0, show_img=True)
        else:
            rgb, depth = self.RS.get_frames()
        self.init_eef_P = self.UR5.get_eef_pose()
        return rgb, depth

    def reset(self, classes, move_ur5=True):
        #if self.default_depth is None:
        #    self.default_depth = np.load('default_depth.npy')

        rgb, depth = self.get_observation(move_ur5)
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
        self.current_classes = classes
        self.current_obs = obs
        return obs

    def check_go(self):
        x = input('go?')
        print(x)
        if x!='y' and x!='go':
            print('exit.')
            exit()

    def move_to_pixel(self, target_position, rot_angle):
        depth = self.current_obs['depth_raw']

        target_pose = inverse_projection(depth, np.array(target_position), self.RS.K_rs, self.RS.D_rs)
        target_pose[2] -= 0.16
        roll, pitch, yaw = np.pi/8, -np.pi/8, 0
        yaw += rot_angle
        yaw %= 2*np.pi
        quat = euler2quat([roll, pitch, yaw])

        placement = form_T(quat2mat(quat), target_pose)
        place_pos, place_quat = self.UR5.get_goal_from_grasp(placement, self.init_eef_P)
        print('original placement:', placement)

        delta_t = np.dot(quat2mat(quat), np.array([[0, 0, -0.1]]).T).T[0]
        placement = form_T(quat2mat(quat), target_pose+delta_t)
        pos1, quat1 = self.UR5.get_goal_from_grasp(placement, self.init_eef_P)
        print('pose 1:', placement)

        delta_t = np.dot(quat2mat(quat), np.array([[0, 0, -0.05]]).T).T[0]
        placement = form_T(quat2mat(quat), target_pose+delta_t)
        pos2, quat2 = self.UR5.get_goal_from_grasp(placement, self.init_eef_P)
        print('pose 2:', placement)

        check_go()
        self.UR5.get_view(pos1, quat1, grasp=0.0)
        check_go()
        self.UR5.get_view(pos2, quat2, grasp=0.0)

        check_go()
        return self.get_observation()

    def step(self, target_object, target_position, rot_angle, stop=True):
        self.check_go()
        rgb = self.current_obs['rgb_raw']
        depth = self.current_obs['depth_raw']
        segmap = self.current_obs['segmentation_raw']

        # 1. Pick up the target object.
        grasps, scores = self.CGN.get_grasps(rgb, depth, segmap, target_object, num_K=1, show_result=False) #True
        #print('grasp:', grasps[0])
        grasp = grasps[0]

        # get delta_t_grasp_to_object_center
        py, px = np.where(segmap==target_object)
        center_position = np.round(np.array([np.mean(px), np.mean(py)])).astype(np.int32)
        center_pose = inverse_projection(depth, center_position, self.RS.K_rs, self.RS.D_rs)
        delta_center = center_pose[:2] - grasp[:2, 3]
        #print("Delta Center:", delta_center)

        #grasp_4dof = project_grasp_4dof(grasp)
        self.pick(grasp, stop=stop, back_to_init=False)

        # 2. Place down at the target position with rotation.
        target_pose = inverse_projection(depth, np.array(target_position), self.RS.K_rs, self.RS.D_rs)
        # get target placement with center offset
        rot_center = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                                [np.sin(rot_angle), np.cos(rot_angle)]])
        target_pose[:2] -= np.dot(rot_center, delta_center)

        roll, pitch, yaw = mat2euler(grasp[:3, :3])
        yaw += rot_angle
        yaw %= 2*np.pi
        quat = euler2quat([roll, pitch, yaw])
        placement = form_T(quat2mat(quat), target_pose)
        #print('placement:', placement)
        self.place(placement, stop=stop)

        return self.reset(self.current_classes)

    def pick(self, grasp, stop=True, back_to_init=True): #, grasp_4dof
        if stop:
            check_go = self.check_go
        else:
            def check_go():
                return None

        target_pose = grasp[:3, 3]
        target_rot = grasp[:3,:3]
        #print('grasp:', grasp)

        delta_t = np.dot(target_rot, np.array([[0, 0, -0.1]]).T).T[0]
        pre_grasp1 = form_T(target_rot, target_pose+delta_t)
        pos1, quat1 = self.UR5.get_goal_from_grasp(pre_grasp1, self.init_eef_P)
        #print('pose 1:', pre_grasp1)

        delta_t = np.dot(target_rot, np.array([[0, 0, -0.05]]).T).T[0]
        pre_grasp2 = form_T(target_rot, target_pose+delta_t)
        pos2, quat2 = self.UR5.get_goal_from_grasp(pre_grasp2, self.init_eef_P)
        #print('pose 2:', pre_grasp2)

        check_go()
        self.UR5.get_view(pos1, quat1)
        check_go()
        self.UR5.get_view(pos2, quat2)
        check_go()
        self.UR5.get_view(grasp=1.0)
        check_go()
        self.UR5.get_view(pos1, quat1, grasp=1.0)
        if back_to_init:
            check_go()
            self.UR5.move_to_joints(self.INIT_JOINTS)
            check_go()
            self.UR5.get_view(self.UR5.ROBOT_INIT_POS, self.UR5.ROBOT_INIT_QUAT, grasp=1.0)
        else:
            check_go()
            self.UR5.get_view(self.UR5.PRE_PLACE_POS, self.UR5.ROBOT_INIT_QUAT, grasp=1.0)


    def place(self, placement, stop=True):
        if stop:
            check_go = self.check_go
        else:
            def check_go():
                return None

        target_pose = placement[:3, 3]
        target_rot = placement[:3,:3]

        delta_t = np.dot(target_rot, np.array([[0, 0, -0.25]]).T).T[0]
        pre_place1 = form_T(target_rot, target_pose+delta_t)
        pos1, quat1 = self.UR5.get_goal_from_grasp(pre_place1, self.init_eef_P)

        delta_t = np.dot(target_rot, np.array([[0, 0, -0.20]]).T).T[0]
        pre_place2 = form_T(target_rot, target_pose+delta_t)
        pos2, quat2 = self.UR5.get_goal_from_grasp(pre_place2, self.init_eef_P)

        #delta_t = np.dot(target_rot, np.array([[0, 0, -0.3]]).T).T[0]
        #pre_place3 = form_T(target_rot, target_pose+delta_t)
        #pos3, quat3 = self.UR5.get_goal_from_grasp(pre_place3, self.init_eef_P)

        check_go()
        self.UR5.get_view(pos1, quat1, grasp=1.0)
        check_go()
        self.UR5.get_view(pos2, quat2, grasp=1.0)
        check_go()
        self.UR5.get_view(grasp=0.0)
        check_go()
        self.UR5.get_view(pos1, quat1, grasp=0.0)
        #check_go()
        #self.UR5.get_view(pos3, quat3, grasp=0.0)
        check_go()
        self.UR5.move_to_joints(self.INIT_JOINTS)
        #check_go()
        #self.UR5.get_view(self.UR5.ROBOT_INIT_POS, grasp=0.0)


if __name__=='__main__':
    env = RealEnvironment(None)
    classes = ['Apple. Cup. Orange. Fruit.']

    obs = env.reset(classes, move_ur5=True)
    plt.imshow(obs['rgb_raw'])#.astype(np.uint8))
    plt.show()
    plt.imshow(obs['depth_raw'])
    plt.show()
    #plt.imshow(obs['depth'])
    #plt.show()
    #plt.imshow(obs['segmentation'])
    #plt.show()

    x = input('where to place?')
    place = [int(p) for p in x.replace(' ', '').split(',')]
    rot_angle = np.pi/3 #0.0
    #env.move_to_pixel(np.array(place), rot_angle)
    env.step(1, np.array(place), rot_angle)
