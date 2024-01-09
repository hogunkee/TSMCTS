import numpy as np
import pybullet as p 
from scene_utils import update_visual_objects, get_rotation
from collect_scenes import TabletopScenes

opt = lambda : None
#opt.nb_objects = 12 #20
#opt.inscene_objects = 4 #5
opt.scene_type = 'line' # 'random' or 'line'
opt.spp = 32 #64 
opt.width = 500
opt.height = 500 
opt.noise = False
#opt.nb_scenes = 1000
#opt.nb_frames = 5
#opt.outf = '/home/gun/ssd/disk/ur5_tidying_data/line-shape/images'
#opt.nb_randomset = 20
opt.dataset = 'train' #'train' or 'test'
opt.objectset = 'ycb' #'pybullet'/'ycb'/'all'
opt.pybullet_object_path = '/home/gun/Desktop/pybullet-URDF-models/urdf_models/models'
opt.ycb_object_path = '/home/gun/ssd/disk/YCB_dataset'

ts = TabletopScenes(opt)
urdf_ids = sorted(ts.urdf_id_names.keys())
for i in range(5):
    urdf_selected = urdf_ids[20 * i:20 * (i+1)]
    ts.spawn_objects(urdf_selected)

    for idx, obj_col_id in enumerate(ts.current_pybullet_ids):
        uid = urdf_selected[idx]
        object_name = ts.urdf_id_names[uid]
        object_type, object_index = uid.split('-')
        pos_new = [ts.xx[idx], ts.yy[idx], 0.25]
        if uid in ts.init_euler:
            print(uid, 'in init_euler.')
            roll, pitch, yaw = np.array(ts.init_euler[uid]) * np.pi / 2
        else:
            print(uid, 'not in init_euler.')
            roll, pitch, yaw = 0, 0, 0
        rot_new = get_rotation(roll, pitch, yaw)
        p.resetBasePositionAndOrientation(obj_col_id, pos_new, rot_new)
        
        for _ in range(200):
            p.stepSimulation()
        nv.ids = update_visual_objects(ts.current_pybullet_ids, "", nv.ids)

        x = input("Press X to exit.")
        if x.lower()=="x":
            exit()
    ts.clear()
ts.close()
