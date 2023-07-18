import blenderproc as bproc
import argparse
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('object', nargs='?', default="examples/basics/camera_object_pose/obj_000004.ply", help="Path to the model file")
parser.add_argument('output_dir', nargs='?', default="examples/basics/camera_object_pose/output", help="Path to where the final files will be saved")
args = parser.parse_args()

num_scenes = 1
cc_textures_path = "resources/cc_textures"

bproc.init()

# load the objects into the scene
obj = bproc.loader.load_obj(args.object)[0]
# Use vertex color for texturing
for mat in obj.get_materials():
    mat.map_vertex_color()
# Set pose of object via local-to-world transformation matrix
obj.set_local2world_mat(
    [[0.331458, -0.9415833, 0.05963787, -0.04474526765165741],
    [-0.6064861, -0.2610635, -0.7510136, 0.08970402424862098],
    [0.7227108, 0.2127592, -0.6575879, 0.6823395750305427],
    [0, 0, 0, 1.0]]
)
# Scale 3D model from mm to m
obj.set_scale([0.001, 0.001, 0.001])
# Set category id which will be used in the BopWriter
obj.set_cp("category_id", 1)

objs = [obj]

# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(200)

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(cc_textures_path)

# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

# Set intrinsics via K matrix
bproc.camera.set_intrinsics_from_K_matrix(
    [[537.4799, 0.0, 318.8965],
     [0.0, 536.1447, 238.3781],
     [0.0, 0.0, 1.0]], 640, 480
)
## Set camera pose via cam-to-world transformation matrix
cam2world = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
# Change coordinate frame of transformation matrix from OpenCV to Blender coordinates
cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world, ["X", "-Y", "-Z"])
bproc.camera.add_camera_pose(cam2world)

# activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_diffuse_color_output()
bproc.renderer.enable_distance_output(activate_antialiasing=True)
#bproc.renderer.enable_motion_blur()
bproc.renderer.enable_normals_output()

for i in range(num_scenes):
    # Sample two light sources
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    light_point.set_location(location)

    # sample CC Texture and assign to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)

    # Sample object poses and check collisions 
    bproc.object.sample_poses(objects_to_sample = objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)
            
    # Define a function that samples the initial pose of a given object above the ground
    def sample_initial_pose(obj: bproc.types.MeshObject):
        obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                    min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
        obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))

    # Sample objects on the given surface
    placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=objs,
                                                        surface=room_planes[0],
                                                        sample_pose_func=sample_initial_pose,
                                                        min_distance=0.01,
                                                        max_distance=0.2)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(objs)

    cam_poses = 0
    while cam_poses < 1:
        # Sample location
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 0.35,
                                radius_max = 1.5,
                                elevation_min = 5,
                                elevation_max = 89)
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(objs)
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # render the whole pipeline
    data = bproc.renderer.render()

    # Write object poses, color and depth in bop format
    bproc.writer.write_hdf5(args.output_dir, data)
    #bproc.writer.write_bop(args.output_dir, [obj], data["depth"], data["normals"], m2mm=True, append_to_existing_output=True)
