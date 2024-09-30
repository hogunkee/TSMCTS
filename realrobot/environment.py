from utils_ur5 import *
from utils_sim2real import inverse_projection
from ellipse import LsqEllipse
from PIL import Image, ImageEnhance

class RealEnvironment:
    def __init__(self, args):
        self.RS = RealSense()
        rospy.sleep(1.0)
        self.CGN = ContactGraspNet(K=self.RS.K_rs)
        self.UR5 = UR5Robot(self.RS)
        self.GSAM = GroundedSAM()
        self.depth_bg = np.load('depth_init.npy')

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
        num_count = 0
        rgb, depth = None, None
        while True:
            num_count += 1
            if move_ur5:
                rgb, depth = self.UR5.get_view(self.UR5.ROBOT_INIT_POS, self.UR5.ROBOT_INIT_QUAT,
                                                grasp=0.0, show_img=True)
            else:
                rgb, depth = self.RS.get_frames()
            if rgb is not None:
                break
            if num_count >= 10:
                break
        self.init_eef_P = self.UR5.get_eef_pose()

        im = Image.fromarray(rgb)
        im = ImageEnhance.Brightness(im).enhance(1.5)
        im = ImageEnhance.Color(im).enhance(1.2)
        im = ImageEnhance.Contrast(im).enhance(0.9)
        rgb = np.array(im)
        return rgb, depth

    def reset(self, classes, move_ur5=True):
        #if self.default_depth is None:
        #    self.default_depth = np.load('default_depth.npy')

        rospy.sleep(1.0)
        rgb, depth = self.get_observation(move_ur5)
        detections = self.GSAM.get_masks(rgb, classes) #, save_image=True)
        class_id = detections.class_id
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
                'segmentation': segmap_pad,
                'class_id': class_id
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

    def step(self, target_object, target_position, rot_angle, stop=True, object_angles=None):
        rgb = self.current_obs['rgb_raw']
        depth = self.current_obs['depth_raw']
        segmap = self.current_obs['segmentation_raw']

        # Grasp Offset according to the Object Category 
        class_id = self.current_obs['class_id'][target_object-1]
        object_class = self.current_classes[class_id]
        if object_class.lower() in ['cup']:
            delta_z = -0.05
        elif object_class.lower() in ['bowl']:
            delta_z = -0.03
        elif object_class.lower() in ['clock', 'teapot']:
            delta_z = -0.015
        else:
            delta_z = 0.

        # Rule-based grasp for low-height objects. #
        depth_delta = (self.depth_bg - depth) * (segmap==target_object)
        depth_delta = depth_delta[depth_delta<0.3]
        if depth_delta.max() < 0.06:
            use_rulebased_grasp = True
            grasps, score = [], []
        else:
            use_rulebased_grasp = False
            grasps, scores = self.CGN.get_grasps(rgb, depth, segmap, target_object, num_K=1, show_result=False) #True

        # 1. Pick up the target object.
        if len(grasps)==0:
            print('Rule-based grasp:')
            py, px = np.where(segmap==target_object)
            X = np.array(list(zip(py, px)))
            #X = np.array(list(zip(*np.where(segmap==target_object))))
            reg = LsqEllipse().fit(X)
            center, height, width, phi = reg.as_parameters()
            #center, width, height, phi = reg.as_parameters()
            print('width:', width)
            print('height:', height)
            #phi += np.pi/2
            if height > width:
                print('height > width. adding 90 degree..')
                phi += np.pi/2
            #if height < width:
            #    phi += np.pi/2
            print('Phi-new:', phi)

            if object_angles is not None:
                phi = object_angles[target_object-1] / 180 * np.pi
                print('Phi-renderer:', phi)

            if max(width, height)<7:
                grasp_type = 'wrong'
                return
            elif width>2*height or height>2*width:
                #elif min(width, height)<7:
                # marker, fork, knife, banana, etc.
                grasp_type = 'center'
            elif width<35 and height<35:
                # small objects: orange, apple, etc
                grasp_type = 'center'
            else:
                # plate, bowl
                grasp_type = 'boundary'
            print('Type:', grasp_type)

            if grasp_type=='center':
                #center = np.array([np.mean(px), (4*py.min() + py.max())/5]).astype(int)
                if depth_delta.max() < 0.04:
                    center = np.array([np.mean(px), np.mean(py)-5]).astype(int)
                    #center = np.array([np.mean(px), np.mean(py)-30]).astype(int)
                else:
                    center = np.array([np.mean(px), np.mean(py)-10]).astype(int)
                    #center = np.array([np.mean(px), np.mean(py)-40]).astype(int)
                #center = np.array([np.mean(px), np.mean(py)]).astype(int)
                translation = inverse_projection(depth, center, self.RS.K_rs, self.RS.D_rs)
            elif grasp_type=='boundary':
                cx = px[py==py.mean().astype(int)].max()
                #cy = (5*py.min() + py.max())/6
                if depth_delta.max() < 0.03:
                    cy = py.mean()-20
                else:
                    cy = py.mean()-40
                #cx = px.mean().astype(int)
                #cx = px.mean().astype(int)
                #cy = py[px==px.mean().astype(int)].max()
                #cx = px[py==py.mean().astype(int)].max()
                #cy = py.mean().astype(int)
                point = np.array([cx, cy]).astype(int)
                print('point:', point)
                translation = inverse_projection(depth, point, self.RS.K_rs, self.RS.D_rs)

            rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            if grasp_type=='center':
                rot = np.dot(quat2mat(euler2quat([0, 0, phi+np.pi])), rot)
                #rot = np.dot(quat2mat(euler2quat([0, 0, -phi+np.pi/2])), rot)
            elif grasp_type=='boundary':
                pass
            grasp = form_T(rot, translation)
            use_rulebased_grasp = True
        else:
            print('GCN grasp:')
            grasp = grasps[0]
            use_rulebased_grasp = False 
        print('grasp:', grasp)

        # get delta_t_grasp_to_object_center
        py, px = np.where(segmap==target_object)
        center_position = np.round(np.array([np.mean(px), np.mean(py)])).astype(np.int32)
        center_pose = inverse_projection(depth, center_position, self.RS.K_rs, self.RS.D_rs)
        delta_center = center_pose[:2] - grasp[:2, 3]
        #print("Delta Center:", delta_center)

        #grasp_4dof = project_grasp_4dof(grasp)
        self.pick(grasp, stop=stop, back_to_init=False, delta_z=delta_z, rulebased=use_rulebased_grasp)

        # 2. Place down at the target position with rotation.
        target_pose = inverse_projection(depth, np.array(target_position), self.RS.K_rs, self.RS.D_rs)
        # get target placement with center offset
        rot_center = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                                [np.sin(rot_angle), np.cos(rot_angle)]])
        target_pose[:2] += np.dot(rot_center, delta_center)
        #target_pose[:2] -= np.dot(rot_center, delta_center)

        roll, pitch, yaw = mat2euler(grasp[:3, :3])
        yaw += -rot_angle # - np.pi/2
        #yaw += rot_angle - np.pi/2
        #yaw %= 2*np.pi
        if yaw > np.pi:
            yaw -= 2*np.pi
        elif yaw < -np.pi:
            yaw += 2*np.pi
        quat = euler2quat([roll, pitch, yaw])
        placement = form_T(quat2mat(quat), target_pose)
        #print('placement:', placement)
        self.place(placement, delta_z=delta_z, stop=stop)

        return self.reset(self.current_classes)

    def step_struct(self, target_object, target_pose, rot_angle, stop=True):
        # target_position, object_angles
        rgb = self.current_obs['rgb_raw']
        depth = self.current_obs['depth_raw']
        segmap = self.current_obs['segmentation_raw']

        # Grasp Offset according to the Object Category 
        class_id = self.current_obs['class_id'][target_object-1]
        object_class = self.current_classes[class_id]
        if object_class.lower() in ['cup']:
            delta_z = -0.05
        elif object_class.lower() in ['bowl']:
            delta_z = -0.03
        elif object_class.lower() in ['clock', 'teapot']:
            delta_z = -0.015
        else:
            delta_z = 0.

        # Rule-based grasp for low-height objects. #
        depth_delta = (self.depth_bg - depth) * (segmap==target_object)
        depth_delta = depth_delta[depth_delta<0.3]
        if depth_delta.max() < 0.06:
            use_rulebased_grasp = True
            grasps, score = [], []
        else:
            use_rulebased_grasp = False
            grasps, scores = self.CGN.get_grasps(rgb, depth, segmap, target_object, num_K=1, show_result=False) #True

        # 1. Pick up the target object.
        if len(grasps)==0:
            print('Rule-based grasp:')
            py, px = np.where(segmap==target_object)
            X = np.array(list(zip(py, px)))
            #X = np.array(list(zip(*np.where(segmap==target_object))))
            reg = LsqEllipse().fit(X)
            center, width, height, phi = reg.as_parameters()
            print('width:', width)
            print('height:', height)
            phi += np.pi/2
            #if height < width:
            #    phi += np.pi/2
            print('Phi-new:', phi)

            if object_angles is not None:
                phi = object_angles[target_object-1] / 180 * np.pi
                print('Phi-renderer:', phi)

            if max(width, height)<7:
                grasp_type = 'wrong'
                return
            elif width>2*height or height>2*width:
                #elif min(width, height)<7:
                # marker, fork, knife, banana, etc.
                grasp_type = 'center'
            elif width<35 and height<35:
                # small objects: orange, apple, etc
                grasp_type = 'center'
            else:
                # plate, bowl
                grasp_type = 'boundary'
            print('Type:', grasp_type)

            if grasp_type=='center':
                #center = np.array([np.mean(px), (4*py.min() + py.max())/5]).astype(int)
                if depth_delta.max() < 0.04:
                    center = np.array([np.mean(px), np.mean(py)-30]).astype(int)
                else:
                    center = np.array([np.mean(px), np.mean(py)-40]).astype(int)
                #center = np.array([np.mean(px), np.mean(py)]).astype(int)
                translation = inverse_projection(depth, center, self.RS.K_rs, self.RS.D_rs)
            elif grasp_type=='boundary':
                cx = px[py==py.mean().astype(int)].max()
                #cy = (5*py.min() + py.max())/6
                if depth_delta.max() < 0.03:
                    cy = py.mean()-20
                else:
                    cy = py.mean()-40
                #cx = px.mean().astype(int)
                #cx = px.mean().astype(int)
                #cy = py[px==px.mean().astype(int)].max()
                #cx = px[py==py.mean().astype(int)].max()
                #cy = py.mean().astype(int)
                point = np.array([cx, cy]).astype(int)
                print('point:', point)
                translation = inverse_projection(depth, point, self.RS.K_rs, self.RS.D_rs)

            rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            if grasp_type=='center':
                rot = np.dot(quat2mat(euler2quat([0, 0, phi+np.pi])), rot)
                #rot = np.dot(quat2mat(euler2quat([0, 0, -phi+np.pi/2])), rot)
            elif grasp_type=='boundary':
                pass
            grasp = form_T(rot, translation)
            use_rulebased_grasp = True
        else:
            print('GCN grasp:')
            grasp = grasps[0]
            use_rulebased_grasp = False 
        print('grasp:', grasp)

        # get delta_t_grasp_to_object_center
        py, px = np.where(segmap==target_object)
        center_position = np.round(np.array([np.mean(px), np.mean(py)])).astype(np.int32)
        center_pose = inverse_projection(depth, center_position, self.RS.K_rs, self.RS.D_rs)
        delta_center = center_pose[:2] - grasp[:2, 3]
        #print("Delta Center:", delta_center)

        #grasp_4dof = project_grasp_4dof(grasp)
        self.pick(grasp, stop=stop, back_to_init=False, delta_z=delta_z, rulebased=use_rulebased_grasp)

        # 2. Place down at the target position with rotation.
        target_pose = inverse_projection(depth, np.array(target_position), self.RS.K_rs, self.RS.D_rs)
        # get target placement with center offset
        rot_center = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                                [np.sin(rot_angle), np.cos(rot_angle)]])
        target_pose[:2] -= np.dot(rot_center, delta_center)

        roll, pitch, yaw = mat2euler(grasp[:3, :3])
        yaw += -rot_angle # - np.pi/2
        #yaw += rot_angle - np.pi/2
        yaw %= 2*np.pi
        quat = euler2quat([roll, pitch, yaw])
        placement = form_T(quat2mat(quat), target_pose)
        #print('placement:', placement)
        self.place(placement, stop=stop)

        return self.reset(self.current_classes)

    def check_picknplace(self, target_object, stop=True):
        if stop:
            check_go = self.check_go
        else:
            def check_go():
                return None
        rgb = self.current_obs['rgb_raw']
        depth = self.current_obs['depth_raw']
        segmap = self.current_obs['segmentation_raw']

        class_id = self.current_obs['class_id'][target_object-1]
        object_class = self.current_classes[class_id]
        if object_class.lower() in ['cup']:
            delta_z = -0.05
        elif object_class.lower() in ['bowl']:
            delta_z = -0.03
        elif object_class.lower() in ['clock', 'teapot']:
            delta_z = -0.015
        else:
            delta_z = 0

        # 1. Find the target object.
        use_rulebased_grasp = False 
        grasps, scores = self.CGN.get_grasps(rgb, depth, segmap, target_object, num_K=1, show_result=False)
        if len(grasps)==0:
            print('Rule-based grasp:')
            py, px = np.where(segmap==target_object)
            X = np.array(list(zip(py, px)))
            #X = np.array(list(zip(*np.where(segmap==target_object))))
            reg = LsqEllipse().fit(X)
            center, width, height, phi = reg.as_parameters()

            print('width:', width)
            print('height:', height)
            if max(width, height)<7:
                grasp_type = 'wrong'
                return
            elif width>2*height or height>2*width:
                #elif min(width, height)<7:
                # marker, fork, knife, banana, etc.
                grasp_type = 'center'
            else:
                # plate
                grasp_type = 'boundary'
            if height > width:
                phi += np.pi/2
            print('Type:', grasp_type)

            if grasp_type=='center':
                center = np.array([np.mean(px), (4*py.min() + py.max())/5]).astype(int)
                #center = np.array([np.mean(px), np.mean(py)]).astype(int)
                translation = inverse_projection(depth, center, self.RS.K_rs, self.RS.D_rs)
            elif grasp_type=='boundary':
                cx = px[py==py.mean().astype(int)].max()
                cy = (5*py.min() + py.max())/6
                #cx = px.mean().astype(int)
                #cy = py[px==px.mean().astype(int)].max()
                #cx = px[py==py.mean().astype(int)].max()
                #cy = py.mean().astype(int)
                point = np.array([cx, cy]).astype(int)
                print('point:', point)
                translation = inverse_projection(depth, point, self.RS.K_rs, self.RS.D_rs)

            rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            if grasp_type=='center':
                rot = np.dot(quat2mat(euler2quat([0, 0, -phi])), rot)
            elif grasp_type=='boundary':
                pass
            grasp = form_T(rot, translation)
            use_rulebased_grasp = True
        else:
            print('GCN grasp:')
            grasp = grasps[0]
            use_rulebased_grasp = False 
        print('grasp:', grasp)

        # 2. Pick
        target_pose = grasp[:3, 3]
        target_rot = grasp[:3,:3]
        #print('grasp:', grasp)

        if use_rulebased_grasp:
            delta_z = -0.12
            delta_t = np.dot(target_rot, np.array([[0, 0, -0.1+delta_z]]).T).T[0]
            pre_grasp1 = form_T(target_rot, target_pose+delta_t)
            pos1, quat1 = self.UR5.get_goal_from_grasp(pre_grasp1, self.init_eef_P)
            if use_rulebased_grasp:
                quat1 = euler2quat([np.pi, 0, mat2euler(quat2mat(quat1))[2]])
            #print('pose 1:', pre_grasp1)

            delta_t = np.dot(target_rot, np.array([[0, 0, -0.05+delta_z]]).T).T[0]
            pre_grasp2 = form_T(target_rot, target_pose+delta_t)
            pos2, quat2 = self.UR5.get_goal_from_grasp(pre_grasp2, self.init_eef_P)
            if use_rulebased_grasp:
                quat2 = euler2quat([np.pi, 0, mat2euler(quat2mat(quat2))[2]])
            #print('pose 2:', pre_grasp2)

        else:
            delta_t = np.dot(target_rot, np.array([[0, 0, -0.1+delta_z]]).T).T[0]
            pre_grasp1 = form_T(target_rot, target_pose+delta_t)
            pos1, quat1 = self.UR5.get_goal_from_grasp(pre_grasp1, self.init_eef_P)
            #print('pose 1:', pre_grasp1)

            delta_t = np.dot(target_rot, np.array([[0, 0, -0.05+delta_z]]).T).T[0]
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

        check_go()
        self.UR5.get_view(self.UR5.PRE_PLACE_POS, [1,0,0,0], grasp=1.0)
        #self.UR5.get_view(self.UR5.PRE_PLACE_POS, self.UR5.ROBOT_INIT_QUAT, grasp=1.0)

        # 3. Place
        check_go()
        self.UR5.get_view(pos2, quat2, grasp=1.0)
        check_go()
        self.UR5.get_view(pos2, quat2, grasp=0.0)
        check_go()
        self.UR5.get_view(grasp=0.0)
        check_go()
        self.UR5.get_view(pos1, quat1, grasp=0.0)
        check_go()
        self.UR5.move_to_joints(self.INIT_JOINTS)

        return
        #return self.reset(self.current_classes)

    def pick(self, grasp, stop=True, back_to_init=True, delta_z=0., rulebased=False):
        if stop:
            check_go = self.check_go
        else:
            def check_go():
                return None

        if rulebased:
            delta_z = -0.12

        target_pose = grasp[:3, 3]
        target_rot = grasp[:3,:3]
        #print('grasp:', grasp)

        delta_t = np.dot(target_rot, np.array([[0, 0, -0.15+delta_z]]).T).T[0]
        pre_grasp1 = form_T(target_rot, target_pose+delta_t)
        pos1, quat1 = self.UR5.get_goal_from_grasp(pre_grasp1, self.init_eef_P)
        if rulebased:
            quat1 = euler2quat([np.pi, 0, mat2euler(quat2mat(quat1))[2]])
        #print('pose 1:', pre_grasp1)

        delta_t = np.dot(target_rot, np.array([[0, 0, -0.05+delta_z]]).T).T[0]
        pre_grasp2 = form_T(target_rot, target_pose+delta_t)
        pos2, quat2 = self.UR5.get_goal_from_grasp(pre_grasp2, self.init_eef_P)
        if rulebased:
            quat2 = euler2quat([np.pi, 0, mat2euler(quat2mat(quat2))[2]])
        #print('pose 2:', pre_grasp2)

        #check_go()
        #self.UR5.get_view(self.UR5.PRE_PLACE_POS, self.UR5.ROBOT_INIT_QUAT) #[1,0,0,0])
        check_go()
        if pos1[0] > 0:
            self.UR5.get_view(self.UR5.PRE_GRASP_POS_1, self.UR5.ROBOT_INIT_QUAT) #[1,0,0,0])
        else:
            self.UR5.get_view(self.UR5.PRE_GRASP_POS_2, self.UR5.ROBOT_INIT_QUAT) #[1,0,0,0])

        if rulebased:
            check_go()
            self.UR5.get_view(pos2+np.array([0, 0, 0.1]), quat2, num_solve=10)
        else:
            check_go()
            self.UR5.get_view(pos1, quat1, num_solve=10)
        check_go()
        self.UR5.get_view(pos2, quat2)
        check_go()
        self.UR5.get_view(grasp=1.0)
        if rulebased:
            check_go()
            self.UR5.get_view(pos2+np.array([0, 0, 0.1]), quat2, grasp=1.0)
        else:
            check_go()
            self.UR5.get_view(pos1, quat1, grasp=1.0)
        check_go()
        self.UR5.get_view(self.UR5.PRE_GRASP_POS_2, self.UR5.ROBOT_INIT_QUAT, grasp=1.0)

        if back_to_init:
            check_go()
            self.UR5.move_to_joints(self.INIT_JOINTS)
            check_go()
            self.UR5.get_view(self.UR5.ROBOT_INIT_POS, self.UR5.ROBOT_INIT_QUAT, grasp=1.0)
        else:
            return
            check_go()
            self.UR5.get_view(self.UR5.PRE_PLACE_POS, [1,0,0,0], grasp=1.0)
            #self.UR5.get_view(self.UR5.PRE_PLACE_POS, self.UR5.ROBOT_INIT_QUAT, grasp=1.0)


    def place(self, placement, delta_z=0., stop=True):
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
        #self.UR5.get_view(pos1, quat1, grasp=1.0)
        self.UR5.get_view(pos2 + np.array([0, 0, 0.1]), quat2, grasp=1.0)
        check_go()
        self.UR5.get_view(pos2 + np.array([0, 0, -delta_z]), quat2, grasp=1.0)
        check_go()
        self.UR5.get_view(grasp=0.0)
        check_go()
        self.UR5.get_view(pos2 + np.array([0, 0, 0.1]), quat2, grasp=0.0)
        #self.UR5.get_view(pos1, quat1, grasp=0.0)
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
