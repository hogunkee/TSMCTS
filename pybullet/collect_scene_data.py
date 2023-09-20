import os 
import nvisii as nv
import random
import colorsys
import subprocess 
import math
import pybullet as p 
import numpy as np
from transform_utils import euler2quat, mat2quat, quat2mat
from scene_utils import init_euler, generate_scene
from scene_utils import get_rotation, get_contact_objects, get_velocity, update_visual_objects

opt = lambda : None
opt.nb_objects = 20
opt.inscene_objects = 5
opt.scene_type = 'random' # 'line'
opt.spp = 32 #64 
opt.width = 500
opt.height = 500 
opt.noise = False
opt.frame_freq = 4 #8
opt.nb_scenes = 2500 #25
opt.nb_frames = 4
opt.outf = 'test_scene' #'scene_data'


# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(opt.outf):
    print(f'folder {opt.outf}/ exists')
else:
    os.mkdir(opt.outf)
    print(f'created folder {opt.outf}/')
# # # # # # # # # # # # # # # # # # # # # # # # #

# show an interactive window, and use "lazy" updates for faster object creation time 
nv.initialize(headless=False, lazy_updates=True)

if not opt.noise is True: 
    nv.enable_denoiser()

# Create a camera
camera = nv.entity.create(
    name = "camera",
    transform = nv.transform.create("camera"),
    camera = nv.camera.create_from_fov(
        name = "camera", 
        field_of_view = 0.85,
        aspect = float(opt.width)/float(opt.height)
    )
)
camera.get_transform().look_at(
    at = (0.5,0,0),
    up = (0,0,1),
    eye = (3, 0, 6), #(4, 0, 6), #(10,0,4),
)
nv.set_camera_entity(camera)

# Setup bullet physics stuff
seconds_per_step = 1.0 / 240.0
frames_per_second = 30.0
physicsClient = p.connect(p.GUI) # non-graphical version
p.setGravity(0,0,-10)

# Lets set the scene

# Change the dome light intensity
nv.set_dome_light_intensity(1.0)

# atmospheric thickness makes the sky go orange, almost like a sunset
nv.set_dome_light_sky(sun_position=(10,10,10), atmosphere_thickness=1.0, saturation=1.0)

# Lets add a sun light
sun = nv.entity.create(
    name = "sun",
    mesh = nv.mesh.create_sphere("sphere"),
    transform = nv.transform.create("sun"),
    light = nv.light.create("sun")
)
sun.get_transform().set_position((10,10,10))
sun.get_light().set_temperature(5780)
sun.get_light().set_intensity(1000)

floor = nv.entity.create(
    name="floor",
    mesh = nv.mesh.create_plane("floor"),
    transform = nv.transform.create("floor"),
    material = nv.material.create("floor")
)
floor.get_transform().set_position((0,0,0))
floor.get_transform().set_scale((6, 6, 6)) #10, 10, 10
floor.get_material().set_roughness(0.1)
floor.get_material().set_base_color((0.8, 0.87, 0.88)) #(0.5,0.5,0.5)

floor_textures = []
texture_files = os.listdir("texture")
texture_files = [f for f in texture_files if f.lower().endswith('.png')]
for i, tf in enumerate(texture_files):
    tex = nv.texture.create_from_file("tex-%d"%i, os.path.join("texture/", tf))
    floor_tex = nv.texture.create_hsv("floor-%d"%i, tex, hue=0, saturation=.5, value=1.0, mix=1.0)
    floor_textures.append((tex, floor_tex))

# Set the collision with the floor mesh
# first lets get the vertices 
vertices = floor.get_mesh().get_vertices()

# get the position of the object
pos = floor.get_transform().get_position()
pos = [pos[0],pos[1],pos[2]]
scale = floor.get_transform().get_scale()
scale = [scale[0],scale[1],scale[2]]
rot = floor.get_transform().get_rotation()
rot = [rot[0],rot[1],rot[2],rot[3]]

# create a collision shape that is a convex hull
obj_col_id = p.createCollisionShape(
    p.GEOM_MESH,
    vertices = vertices,
    meshScale = scale,
)

# create a body without mass so it is static
p.createMultiBody(
    baseCollisionShapeIndex = obj_col_id,
    basePosition = pos,
    baseOrientation= rot,
)    

# lets create a bunch of objects 
object_path = '/home/gun/Desktop/pybullet-URDF-models/urdf_models/models'
object_names = sorted([m for m in os.listdir(object_path) if os.path.isdir(os.path.join(object_path, m))])
urdf_id_names = dict(zip(range(len(object_names)), object_names))
print(len(urdf_id_names), 'objects can be loaded.')
urdf_selected = np.random.choice(list(urdf_id_names.keys()), opt.nb_objects, replace=False)

x = np.linspace(-4, 4, 6)
y = np.linspace(-4, 4, 6)
xx, yy = np.meshgrid(x, y, sparse=False)
xx = xx.reshape(-1)
yy = yy.reshape(-1)

pybullet_ids = []
for idx, urdf_id in enumerate(urdf_selected):
    object_name = urdf_id_names[urdf_id]
    urdf_path = os.path.join(object_path, object_name, 'model.urdf')
    obj_col_id = p.loadURDF(urdf_path, [xx[idx], yy[idx], 0.5], globalScaling=5.)
    pybullet_ids.append(obj_col_id)
nv.ids = update_visual_objects(pybullet_ids, "")

#threshold_pose = 0.05
threshold_rotation = 0.01
threshold_linear = 0.003
threshold_angular = 0.003
pre_selected_objects = pybullet_ids 

# Lets run the simulation for a few steps. 
num_exist_frames = len([f for f in os.listdir(f"{opt.outf}") if '.png' in f])
ns = 0
while ns < opt.nb_scenes:
    # set floor material #
    roughness = random.uniform(0.1, 0.5)
    floor.get_material().clear_base_color_texture()
    floor.get_material().set_roughness(roughness)

    f_cidx = np.random.choice(len(floor_textures))
    tex, floor_tex = floor_textures[f_cidx]
    floor.get_material().set_base_color_texture(floor_tex)
    floor.get_material().set_roughness_texture(tex)

    # set objects #
    selected_objects = np.random.choice(pybullet_ids, opt.inscene_objects, replace=False)
    for idx, urdf_id in enumerate(urdf_selected):
        obj_col_id = pybullet_ids[idx]
        if obj_col_id in pre_selected_objects:
            pos_hidden = [xx[idx], yy[idx], -1]
            p.resetBasePositionAndOrientation(obj_col_id, pos_hidden, [0, 0, 0, 1])

        init_positions = generate_scene(opt.scene_type, opt.inscene_objects)
        for i, obj_col_id in enumerate(selected_objects):
            pos_sel = init_positions[i]
            #pos_sel = 4*(np.random.rand(3) - 0.5)
            #pos_sel[2] = 0.6
            roll, pitch, yaw = 0, 0, 0
            if urdf_id in init_euler:
                roll, pitch, yaw = np.array(init_euler[urdf_id]) * np.pi / 2
            rot = get_rotation(roll, pitch, yaw)
            p.resetBasePositionAndOrientation(obj_col_id, pos_sel, rot)

    for j in range(2000):
        p.stepSimulation()
        vel_linear, vel_rot = get_velocity(selected_objects)
        stop_linear = (np.linalg.norm(vel_linear) < threshold_linear)
        stop_rotation = (np.linalg.norm(vel_rot) < threshold_angular)
        if j%10==0:
            if stop_linear and stop_rotation:
                break

    if j==1999: 
        pre_selected_objects = selected_objects
        continue
    nv.ids = update_visual_objects(pybullet_ids, "", nv.ids)

    obj_to_repose = []
    count_scene_init = 0
    while True:
        # re-positioning objects #
        for idx, urdf_id in enumerate(urdf_selected):
            obj_col_id = pybullet_ids[idx]
            if obj_col_id not in selected_objects:
                continue
            if obj_col_id in obj_to_repose:
                pos_repose = generate_scene('random', 1)[0]
                #pos_repose = 4*(np.random.rand(3) - 0.5)
                #pos_repose[2] = 0.6
                roll, pitch, yaw = 0, 0, 0
                if urdf_id in init_euler:
                    roll, pitch, yaw = np.array(init_euler[urdf_id]) * np.pi / 2
                rot = get_rotation(roll, pitch, yaw)
                p.resetBasePositionAndOrientation(obj_col_id, pos_repose, rot)
        # check collisions #
        obj_to_repose = []
        for idx, urdf_id in enumerate(urdf_selected):
            obj_col_id = pybullet_ids[idx]
            if obj_col_id not in selected_objects:
                continue
            pos, rot = p.getBasePositionAndOrientation(obj_col_id)
            roll, pitch, yaw = 0, 0, 0
            if urdf_id in init_euler:
                roll, pitch, yaw = np.array(init_euler[urdf_id]) * np.pi / 2
            rot_init = get_rotation(roll, pitch, yaw)
            rot_diff = np.linalg.norm(np.array(rot) - np.array(rot_init))
            if rot_diff > threshold_rotation:
                obj_to_repose.append(obj_col_id)
        if len(obj_to_repose)==0:
            break
        count_scene_init += 1
        if count_scene_init > 10:
            break
    # if fails to initialize the scene, skip the current objects set
    if count_scene_init > 10:
        continue

    #for nf in range(int(opt.nb_frames)):
    nf = 0
    targets = np.random.choice(selected_objects, opt.nb_frames, replace=False)
    while nf < int(opt.nb_frames):
        # save current poses & rots #
        pos_saved, rot_saved = {}, {}
        for idx, urdf_id in enumerate(urdf_selected):
            obj_col_id = pybullet_ids[idx]
            if obj_col_id not in selected_objects:
                continue

            pos, rot = p.getBasePositionAndOrientation(obj_col_id)
            pos_saved[obj_col_id] = pos
            rot_saved[obj_col_id] = rot

        # set poses & rots #
        target = targets[nf]
        for idx, urdf_id in enumerate(urdf_selected):
            obj_col_id = pybullet_ids[idx]
            if obj_col_id != target:
                continue

            flag_collision = True
            count_scene_repose = 0
            while flag_collision:
                # get the pose of the objects
                pos, rot = p.getBasePositionAndOrientation(obj_col_id)
                collisions_before = get_contact_objects()

                pos_new = 4*(np.random.rand(3) - 0.5)
                pos_new[2] = 0.6
                roll, pitch, yaw = 0, 0, 0
                if urdf_id in init_euler:
                    roll, pitch, yaw = np.array(init_euler[urdf_id]) * np.pi / 2
                rot = get_rotation(roll, pitch, yaw)
                p.resetBasePositionAndOrientation(obj_col_id, pos_new, rot)
                collisions_after = set()
                for _ in range(200):
                    p.stepSimulation()
                    collisions_after = collisions_after.union(get_contact_objects())

                collisions_new = collisions_after - collisions_before
                if len(collisions_new) > 0:
                    flag_collision = True

                    # reset non-target objects
                    obj_to_reset = set()
                    for collision in collisions_new:
                        obj1, obj2 = collision
                        obj_to_reset.add(obj1)
                        obj_to_reset.add(obj2)
                    obj_to_reset = obj_to_reset - set([obj_col_id])
                    for reset_col_id in obj_to_reset:
                        p.resetBasePositionAndOrientation(reset_col_id, pos_saved[reset_col_id], rot_saved[reset_col_id])
                else:
                    flag_collision = False
                    pos, rot = p.getBasePositionAndOrientation(obj_col_id)
                    pos_saved[obj_col_id] = pos
                    rot_saved[obj_col_id] = rot
                count_scene_repose += 1
                if count_scene_repose > 10:
                    break
            if count_scene_repose > 10:
                continue

        for j in range(2000):
            p.stepSimulation()
            vel_linear, vel_rot = get_velocity(selected_objects)
            stop_linear = (np.linalg.norm(vel_linear) < threshold_linear)
            stop_rotation = (np.linalg.norm(vel_rot) < threshold_angular)
            if j%10==0:
                if stop_linear and stop_rotation:
                    break
        nv.ids = update_visual_objects(pybullet_ids, "", nv.ids)

        print(f'rendering scene {str(ns).zfill(5)}-{str(nf)}', end='\r')
        nv.render_to_file(
            width=int(opt.width), 
            height=int(opt.height), 
            samples_per_pixel=int(opt.spp),
            file_path=f"{opt.outf}/{str(num_exist_frames + ns * opt.nb_frames + nf).zfill(5)}.png"
        )
        nf += 1
    pre_selected_objects = selected_objects
    ns += 1

p.disconnect()
nv.deinitialize()

