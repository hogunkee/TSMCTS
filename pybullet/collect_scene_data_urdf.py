import os 
import nvisii as nv
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
    rot = nv.normalize(nv.quat(w, x, y, z))
    return rot

def set_object_pose(ids, pos, rot=None):
    if rot is None:
        _, rot = p.getBasePositionAndOrientation(ids['pybullet_id'])
    p.resetBasePositionAndOrientation(ids['pybullet_id'], pos, rot)

    # get the nv.entity for that object
    obj_entity = nv.entity.get(ids['nv.id'])
    obj_entity.get_transform().set_position(pos)

    if rot is not None:
        # nv.quat expects w as the first argument
        obj_entity.get_transform().set_rotation(rot)
    return

def sync_object_poses(object_ids):
    for ids in object_ids:
        pos, rot = p.getBasePositionAndOrientation(ids['pybullet_id'])
        obj_entity = nv.entity.get(ids['nv.id'])
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
    for pid in object_ids:
        vel_linear, vel_rot = p.getBaseVelocity(pid)
        velocities_linear.append(vel_linear)
        velocities_rotation.append(vel_rot)
    return velocities_linear, velocities_rotation

def update_visual_objects(object_ids, pkg_path, nv_objects=None):
    # object ids are in pybullet engine
    # pkg_path is for loading the object geometries
    # nv_objects refers to the already entities loaded, otherwise it is going 
    # to load the geometries and create entities. 
    if nv_objects is None:
        nv_objects = { }
    for object_id in object_ids:
        for idx, visual in enumerate(p.getVisualShapeData(object_id)):
            # Extract visual data from pybullet
            objectUniqueId = visual[0]
            linkIndex = visual[1]
            visualGeometryType = visual[2]
            dimensions = visual[3]
            meshAssetFileName = visual[4]
            local_visual_frame_position = visual[5]
            local_visual_frame_orientation = visual[6]
            rgbaColor = visual[7]

            if linkIndex == -1:
                dynamics_info = p.getDynamicsInfo(object_id,-1)
                inertial_frame_position = dynamics_info[3]
                inertial_frame_orientation = dynamics_info[4]
                base_state = p.getBasePositionAndOrientation(objectUniqueId)
                world_link_frame_position = base_state[0]
                world_link_frame_orientation = base_state[1]    
                m1 = nv.translate(nv.mat4(1), nv.vec3(inertial_frame_position[0], inertial_frame_position[1], inertial_frame_position[2]))
                m1 = m1 * nv.mat4_cast(nv.quat(inertial_frame_orientation[3], inertial_frame_orientation[0], inertial_frame_orientation[1], inertial_frame_orientation[2]))
                m2 = nv.translate(nv.mat4(1), nv.vec3(world_link_frame_position[0], world_link_frame_position[1], world_link_frame_position[2]))
                m2 = m2 * nv.mat4_cast(nv.quat(world_link_frame_orientation[3], world_link_frame_orientation[0], world_link_frame_orientation[1], world_link_frame_orientation[2]))
                m = nv.inverse(m1) * m2
                q = nv.quat_cast(m)
                world_link_frame_position = m[3]
                world_link_frame_orientation = q
            else:
                linkState = p.getLinkState(objectUniqueId, linkIndex)
                world_link_frame_position = linkState[4]
                world_link_frame_orientation = linkState[5]
            
            # Name to use for components
            object_name = f"{objectUniqueId}_{linkIndex}_{idx}"

            meshAssetFileName = meshAssetFileName.decode('UTF-8')
            if object_name not in nv_objects:
                # Create mesh component if not yet made
                if visualGeometryType == p.GEOM_MESH:
                    try:
                        #print(meshAssetFileName)
                        nv_objects[object_name] = nv.import_scene(
                            os.path.join(pkg_path, meshAssetFileName)
                        )
                    except Exception as e:
                        print(e)
                elif visualGeometryType == p.GEOM_BOX:
                    assert len(meshAssetFileName) == 0
                    nv_objects[object_name] = nv.entity.create(
                        name=object_name,
                        mesh=nv.mesh.create_box(
                            name=object_name,
                            # half dim in nv.v.s. pybullet
                            size=nv.vec3(dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2)
                        ),
                        transform=nv.transform.create(object_name),
                        material=nv.material.create(object_name),
                    )
                elif visualGeometryType == p.GEOM_CYLINDER:
                    assert len(meshAssetFileName) == 0
                    length = dimensions[0]
                    radius = dimensions[1]
                    nv_objects[object_name] = nv.entity.create(
                        name=object_name,
                        mesh=nv.mesh.create_cylinder(
                            name=object_name,
                            radius=radius,
                            size=length / 2,    # size in nv.is half of the length in pybullet
                        ),
                        transform=nv.transform.create(object_name),
                        material=nv.material.create(object_name),
                    )
                elif visualGeometryType == p.GEOM_SPHERE:
                    assert len(meshAssetFileName) == 0
                    nv_objects[object_name] = nv.entity.create(
                        name=object_name,
                        mesh=nv.mesh.create_sphere(
                            name=object_name,
                            radius=dimensions[0],
                        ),
                        transform=nv.transform.create(object_name),
                        material=nv.material.create(object_name),
                    )
                else:
                    # other primitive shapes currently not supported
                    continue

            if object_name not in nv_objects: continue

            # Link transform
            m1 = nv.translate(nv.mat4(1), nv.vec3(world_link_frame_position[0], world_link_frame_position[1], world_link_frame_position[2]))
            m1 = m1 * nv.mat4_cast(nv.quat(world_link_frame_orientation[3], world_link_frame_orientation[0], world_link_frame_orientation[1], world_link_frame_orientation[2]))

            # Visual frame transform
            m2 = nv.translate(nv.mat4(1), nv.vec3(local_visual_frame_position[0], local_visual_frame_position[1], local_visual_frame_position[2]))
            m2 = m2 * nv.mat4_cast(nv.quat(local_visual_frame_orientation[3], local_visual_frame_orientation[0], local_visual_frame_orientation[1], local_visual_frame_orientation[2]))

            # import scene directly with mesh files
            if isinstance(nv_objects[object_name], nv.scene):
                # Set root transform of visual objects collection to above transform
                nv_objects[object_name].transforms[0].set_transform(m1 * m2)
                nv_objects[object_name].transforms[0].set_scale(dimensions)

                for m in nv_objects[object_name].materials:
                    m.set_base_color((rgbaColor[0] ** 2.2, rgbaColor[1] ** 2.2, rgbaColor[2] ** 2.2))
            # for entities created for primitive shapes
            else:
                nv_objects[object_name].get_transform().set_transform(m1 * m2)
                nv_objects[object_name].get_material().set_base_color(
                    (
                        rgbaColor[0] ** 2.2,
                        rgbaColor[1] ** 2.2,
                        rgbaColor[2] ** 2.2,
                    )
                )
            # print(visualGeometryType)
    return nv_objects

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
    at = (0,0,0),
    up = (0,0,1),
    eye = (2, 0, 4), #(4, 0, 6), #(10,0,4),
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
texture_files = os.listdir("texture")
texture_files = [f for f in texture_files if f.lower().endswith('.png')]
#texture_files = ["Wood_BaseColor.jpg", "WoodFloor_BaseColor.jpg", "WoodPanel.png"]
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
object_names = [m for m in os.listdir(object_path) if os.path.isdir(os.path.join(object_path, m))]
object_names = np.random.choice(object_names, opt.nb_objects, replace=False)

x = np.linspace(-3, 3, 7)
y = np.linspace(-3, 3, 7)
xx, yy = np.meshgrid(x, y, sparse=False)
xx = xx.reshape(-1)
yy = yy.reshape(-1)

pybullet_ids = []
for object_name in object_names:
    urdf_path = os.path.join(object_path, object_name, 'model.urdf')
    obj_col_id = p.loadURDF(urdf_path, [xx[idx], yy[idx], 0.5], globalScaling=5.)
    pybullet_ids.append(obj_col_id)
nv.ids = update_visual_objects(pybullet_ids, "")

threshold_linear = 0.003
threshold_rotation = 0.003
pre_selected_objects = pybullet_ids 

# Lets run the simulation for a few steps. 
num_exist_frames = len([f for f in os.listdir(f"{opt.outf}") if '.png' in f])
for ns in range (int(opt.nb_scenes)):
    # set objects #
    selected_objects = np.random.choice(pybullet_ids, opt.inscene_objects, replace=False)
    for idx, obj in enumerate(pre_selected_objects):
        pos_hidden = [xx[idx], yy[idx], -1]
        p.resetBasePositionAndOrientation(obj, pos_hidden, [0, 0, 0, 1])
        #set_object_pose(obj, pos_hidden, rot=[1, 0, 0, 0])

    for idx, obj in enumerate(selected_objects):
        pos_sel = 5*(np.random.rand(3) - 0.5)
        pos_sel[2] = 1
        p.resetBasePositionAndOrientation(obj, pos_sel, [0, 0, 0, 1])
        #set_object_pose(obj, pos_sel, rot=[1, 0, 0, 0])

    roughness = random.uniform(0.1, 0.5)
    floor.get_material().clear_base_color_texture()
    floor.get_material().set_roughness(roughness)
    f_cidx = np.random.choice(len(floor_textures))
    tex, floor_tex = floor_textures[f_cidx]
    floor.get_material().set_base_color_texture(floor_tex)
    floor.get_material().set_roughness_texture(tex)
    #floor_cidx = np.random.choice(len(floor_colors)+len(floor_textures))
    #if floor_cidx < len(floor_colors):
    #    floor.get_material().set_base_color(floor_colors[floor_cidx])
    #else:
    #    f_cidx = floor_cidx - len(floor_colors)
    #    tex, floor_tex = floor_textures[f_cidx]
    #    floor.get_material().set_base_color_texture(floor_tex)
    #    floor.get_material().set_roughness_texture(tex)


    for nf in range(int(opt.nb_frames)):
        # set poses & rots #
        # get frames #
        # Lets update the pose of the objects in nv.
        targets = np.random.choice(selected_objects, 1, replace=False)
        for pybullet_id in targets:
            flag_collision = True
            while flag_collision:
                # get the pose of the objects
                pos, rot = p.getBasePositionAndOrientation(pybullet_id)
                print(rot)
                collisions_before = get_contact_objects()

                pos_new = 5*(np.random.rand(3) - 0.5)
                pos_new[2] = 1
                p.resetBasePositionAndOrientation(pybullet_id, pos_new, [0, 0, 0, 1])
                #set_object_pose(ids, pos_new, rot=[1, 0, 0, 0])
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
        nv.ids = update_visual_objects(pybullet_ids, "", nv.ids)
        #sync_object_poses(selected_objects)

        print(f'rendering frame {str(i).zfill(5)}/{str(opt.nb_frames).zfill(5)}')
        nv.render_to_file(
            width=int(opt.width), 
            height=int(opt.height), 
            samples_per_pixel=int(opt.spp),
            file_path=f"{opt.outf}/{str(num_exist_frames + ns * opt.nb_frames + nf).zfill(5)}.png"
        )
    pre_selected_objects = selected_objects

p.disconnect()
nv.deinitialize()

#subprocess.call(['ffmpeg', '-y', '-framerate', '30', '-i', r"%05d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '../output.mp4'], cwd=os.path.realpath(opt.outf))
