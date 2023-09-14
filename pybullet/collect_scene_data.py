import os 
import nvisii
import random
import colorsys
import subprocess 
import math
import pybullet as p 
import numpy as np
from transform_utils import euler2quat

opt = lambda : None
opt.nb_objects = 10 #30 #50
opt.inscene_objects = 5
opt.spp = 32 #64 
opt.width = 500
opt.height = 500 
opt.noise = False
opt.frame_freq = 4 #8
opt.nb_scenes = 2500 #25
opt.nb_frames = 4
opt.outf = 'scene_data'

def get_rotation(roll, pitch, yaw):
    euler = roll, pitch, yaw
    x, y, z, w = euler2quat(euler)
    rot = nvisii.normalize(nvisii.quat(w, x, y, z))
    return rot

def set_object_pose(ids, pos, rot=None):
    if rot is None:
        _, rot = p.getBasePositionAndOrientation(ids['pybullet_id'])
    p.resetBasePositionAndOrientation(ids['pybullet_id'], pos, rot)

    # get the nvisii entity for that object
    obj_entity = nvisii.entity.get(ids['nvisii_id'])
    obj_entity.get_transform().set_position(pos)

    if rot is not None:
        # nvisii quat expects w as the first argument
        obj_entity.get_transform().set_rotation(rot)
    return

def sync_object_poses(object_ids):
    for ids in object_ids:
        pos, rot = p.getBasePositionAndOrientation(ids['pybullet_id'])
        obj_entity = nvisii.entity.get(ids['nvisii_id'])
        obj_entity.get_transform().set_position(pos)
        obj_entity.get_transform().set_rotation(rot)
    return

def get_contact_objects():
    contact_pairs = set()
    for contact in p.getContactPoints():
        body_A = contact[1]
        body_B = contact[2]
        contact_pairs.add(tuple(sorted((body_A, body_B))))
    collisions = set()
    for cp in contact_pairs:
        if cp[0] == 0:
            continue
        collisions.add(cp)
    return collisions

def get_velocity(object_ids):
    velocities_linear = []
    velocities_rotation = []
    for ids in object_ids:
        vel_linear, vel_rot = p.getBaseVelocity(ids['pybullet_id'])
        velocities_linear.append(vel_linear)
        velocities_rotation.append(vel_rot)
    return velocities_linear, velocities_rotation

# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(opt.outf):
    print(f'folder {opt.outf}/ exists')
else:
    os.mkdir(opt.outf)
    print(f'created folder {opt.outf}/')
# # # # # # # # # # # # # # # # # # # # # # # # #

# show an interactive window, and use "lazy" updates for faster object creation time 
nvisii.initialize(headless=False, lazy_updates=True)

if not opt.noise is True: 
    nvisii.enable_denoiser()

# Create a camera
camera = nvisii.entity.create(
    name = "camera",
    transform = nvisii.transform.create("camera"),
    camera = nvisii.camera.create_from_fov(
        name = "camera", 
        field_of_view = 0.85,
        aspect = float(opt.width)/float(opt.height)
    )
)
camera.get_transform().look_at(
    at = (0,0,0),
    up = (0,0,1),
    eye = (2, 0, 4), #(4, 0, 6), #(10,0,4),
)
nvisii.set_camera_entity(camera)

# Setup bullet physics stuff
seconds_per_step = 1.0 / 240.0
frames_per_second = 30.0
physicsClient = p.connect(p.GUI) # non-graphical version
p.setGravity(0,0,-10)

# Lets set the scene

# Change the dome light intensity
nvisii.set_dome_light_intensity(1.0)

# atmospheric thickness makes the sky go orange, almost like a sunset
nvisii.set_dome_light_sky(sun_position=(10,10,10), atmosphere_thickness=1.0, saturation=1.0)

# Lets add a sun light
sun = nvisii.entity.create(
    name = "sun",
    mesh = nvisii.mesh.create_sphere("sphere"),
    transform = nvisii.transform.create("sun"),
    light = nvisii.light.create("sun")
)
sun.get_transform().set_position((10,10,10))
sun.get_light().set_temperature(5780)
sun.get_light().set_intensity(1000)

floor = nvisii.entity.create(
    name="floor",
    mesh = nvisii.mesh.create_plane("floor"),
    transform = nvisii.transform.create("floor"),
    material = nvisii.material.create("floor")
)
floor.get_transform().set_position((0,0,0))
floor.get_transform().set_scale((4, 4, 4)) #10, 10, 10
floor.get_material().set_roughness(0.1)
floor.get_material().set_base_color((0.8, 0.87, 0.88)) #(0.5,0.5,0.5)

floor_colors = np.array([
    [139,90,43],
    [255,165,79],
    [160,82,45],
    [205,133,0],
    [139,69,19],
    [204,222,224]
    ]) / 255.
floor_textures = []
texture_files = ["Wood_BaseColor.jpg", "WoodFloor_BaseColor.jpg", "WoodPanel.png"]
for i, tf in enumerate(texture_files):
    tex = nvisii.texture.create_from_file("tex-%d"%i, os.path.join("texture/", tf))
    floor_tex = nvisii.texture.create_hsv("floor-%d"%i, tex, hue=0, saturation=.5, value=1.0, mix=1.0)
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
object_names = [m for m in os.listdir(object_path) if os.path.isdir(os.path.join(object_path, m))]
object_names = np.random.choice(object_names, opt.nb_objects, replace=False)

meshes = []
list_vertices = []
list_indices = []
for object_name in object_names:
    mesh = nvisii.mesh.create_from_file(f"mesh_{object_name}", os.path.join(object_path, object_name, 'collision.obj'))
    meshes.append(mesh)

    # set up for pybullet - here we will use indices for 
    # objects with holes 
    vertices = mesh.get_vertices()
    indices = mesh.get_triangle_indices()
    list_vertices.append(vertices)
    list_indices.append(indices)

ids_pybullet_and_nvisii_names = []

mesh_indices = np.arange(len(meshes))
#mesh_indices = np.random.choice(np.arange(len(meshes)), opt.nb_objects, replace=False)
for i, idx in enumerate(mesh_indices):
    name = f"mesh_{i}"
    obj = nvisii.entity.create(
                name = name,
                transform = nvisii.transform.create(name),
                material = nvisii.material.create(name)
            )
    obj.set_mesh(meshes[idx])

    # transforms
    pos = nvisii.vec3(
        random.uniform(-2, 2), #(-4,4),
        random.uniform(-2, 2), #(-4,4),
        random.uniform(1, 1.2) #(2,5)
    )
    rot = get_rotation(0.0, 0.0, 0.0)
    #rot = nvisii.normalize(nvisii.quat(
    #    random.uniform(-1,1),
    #    random.uniform(-1,1),
    #    random.uniform(-1,1),
    #    random.uniform(-1,1),
    #))
    s = random.uniform(3, 5) #(0.2,0.5)
    scale = (s,s,s)

    obj.get_transform().set_position(pos)
    obj.get_transform().set_rotation(rot)
    obj.get_transform().set_scale(scale)

    # pybullet setup 
    pos = [pos[0],pos[1],pos[2]]
    rot = [rot[0],rot[1],rot[2],rot[3]]
    scale = [scale[0],scale[1],scale[2]]

    obj_col_id = p.createCollisionShape(
        p.GEOM_MESH,
        vertices = list_vertices[idx],
        meshScale = scale,
        # if you have static object like a bowl
        # this allows you to have concave objects, but 
        # for non concave object, using indices is 
        # suboptimal, you can uncomment if you want to test
        #indices =  list_indices[idx],  
    )
    
    p.createMultiBody(
        baseCollisionShapeIndex = obj_col_id,
        basePosition = pos,
        baseOrientation= rot,
        baseMass = random.uniform(0.5,2)
    )       

    # to keep track of the ids and names 
    ids_pybullet_and_nvisii_names.append(
        {
            "pybullet_id":obj_col_id, 
            "nvisii_id":name
        }
    )

    # Material setting
    rgb = colorsys.hsv_to_rgb(
        random.uniform(0,1),
        random.uniform(0.7,1),
        random.uniform(0.7,1)
    )

    obj.get_material().set_base_color(rgb)

    obj_mat = obj.get_material()
    r = random.randint(0,2)

    # This is a simple logic for more natural random materials, e.g.,  
    # mirror or glass like objects
    if r == 0:  
        # Plastic / mat
        obj_mat.set_metallic(0)  # should 0 or 1      
        obj_mat.set_transmission(0)  # should 0 or 1      
        obj_mat.set_roughness(random.uniform(0,1)) # default is 1  
    if r == 1:  
        # metallic
        obj_mat.set_metallic(random.uniform(0.9,1))  # should 0 or 1      
        obj_mat.set_transmission(0)  # should 0 or 1      
    if r == 2:  
        # glass
        obj_mat.set_metallic(0)  # should 0 or 1      
        obj_mat.set_transmission(random.uniform(0.9,1))  # should 0 or 1      

    if r > 0: # for metallic and glass
        r2 = random.randint(0,1)
        if r2 == 1: 
            obj_mat.set_roughness(random.uniform(0,.1)) # default is 1  
        else:
            obj_mat.set_roughness(random.uniform(0.9,1)) # default is 1  


threshold_linear = 0.003
threshold_rotation = 0.003
pre_selected_objects = ids_pybullet_and_nvisii_names

x = np.linspace(-1.5, 1.5, 7)
y = np.linspace(-1.5, 1.5, 7)
xx, yy = np.meshgrid(x, y, sparse=False)
xx = xx.reshape(-1)
yy = yy.reshape(-1)

# Lets run the simulation for a few steps. 
num_exist_frames = len([f for f in os.listdir(f"{opt.outf}") if '.png' in f])
for ns in range (int(opt.nb_scenes)):
    # set objects #
    selected_objects = np.random.choice(ids_pybullet_and_nvisii_names, opt.inscene_objects, replace=False)
    for idx, obj in enumerate(pre_selected_objects):
        pos_hidden = [xx[idx], yy[idx], -1]
        set_object_pose(obj, pos_hidden, rot=[1, 0, 0, 0])

    for idx, obj in enumerate(selected_objects):
        pos_sel = 2*(np.random.rand(3) - 0.5)
        pos_sel[2] = 0.5
        set_object_pose(obj, pos_sel, rot=[1, 0, 0, 0])

    # set texture of objects #
    for ids in ids_pybullet_and_nvisii_names:
        # get the nvisii entity for that object
        obj_entity = nvisii.entity.get(ids['nvisii_id'])
        # Material setting
        rgb = colorsys.hsv_to_rgb(
            random.uniform(0,1),
            random.uniform(0.7,1),
            random.uniform(0.7,1)
        )
        obj_entity.get_material().set_base_color(rgb)

    roughness = random.uniform(0.1, 0.5)
    floor.get_material().clear_base_color_texture()
    floor.get_material().set_roughness(roughness)
    floor_cidx = np.random.choice(len(floor_colors)+len(floor_textures))
    if floor_cidx < len(floor_colors):
        floor.get_material().set_base_color(floor_colors[floor_cidx])
    else:
        f_cidx = floor_cidx - len(floor_colors)
        tex, floor_tex = floor_textures[f_cidx]
        floor.get_material().set_base_color_texture(floor_tex)
        floor.get_material().set_roughness_texture(tex)


    for nf in range(int(opt.nb_frames)):
        # set poses & rots #
        # get frames #
        # Lets update the pose of the objects in nvisii 
        targets = np.random.choice(selected_objects, 1, replace=False)
        for ids in targets: #ids_pybullet_and_nvisii_names:
            flag_collision = True
            while flag_collision:
                # get the pose of the objects
                pos, rot = p.getBasePositionAndOrientation(ids['pybullet_id'])
                print(rot)
                collisions_before = get_contact_objects()

                pos_new = 2*(np.random.rand(3) - 0.5)
                pos_new[2] = 0.5
                set_object_pose(ids, pos_new, rot=[1, 0, 0, 0])
                collisions_after = get_contact_objects()

                collisions_new = collisions_after - collisions_before
                if len(collisions_new) > 0:
                    print('collision')
                    flag_collision = True
                else:
                    flag_collision = False

        #steps_per_frame = math.ceil( 1.0 / (seconds_per_step * frames_per_second) )
        for j in range(2000):
            p.stepSimulation()
            vel_linear, vel_rot = get_velocity(selected_objects)
            stop_linear = (np.linalg.norm(vel_linear) < threshold_linear)
            stop_rotation = (np.linalg.norm(vel_rot) < threshold_rotation)
            if j%10==0:
                if stop_linear and stop_rotation:
                    break
        sync_object_poses(selected_objects)

        print(f'rendering frame {str(i).zfill(5)}/{str(opt.nb_frames).zfill(5)}')
        nvisii.render_to_file(
            width=int(opt.width), 
            height=int(opt.height), 
            samples_per_pixel=int(opt.spp),
            file_path=f"{opt.outf}/{str(num_exist_frames + ns * opt.nb_frames + nf).zfill(5)}.png"
        )
    pre_selected_objects = selected_objects

p.disconnect()
nvisii.deinitialize()

subprocess.call(['ffmpeg', '-y', '-framerate', '30', '-i', r"%05d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '../output.mp4'], cwd=os.path.realpath(opt.outf))
