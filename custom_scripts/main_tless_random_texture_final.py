# This is a modified version of:
# https://github.com/DLR-RM/BlenderProc/blob/main/examples/datasets/bop_challenge/main_tless_random.py

import blenderproc as bproc
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', help="Path to the bop datasets parent directory")
parser.add_argument('cc_textures_path', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', help="Path to where the final files will be saved ")
parser.add_argument('--variant', choices=['A', 'B'], required=True, help="Texture variant label (A or B)")
parser.add_argument('--scene_seed', type=int, required=True, help="Seed to fix the scene layout")
parser.add_argument('--views', type=int, default=25, help="Number of camera views per scene")
args = parser.parse_args()

np.random.seed(args.scene_seed)

bproc.init()

# load bop objects into the scene
target_bop_objs = bproc.loader.load_bop_objs(
    bop_dataset_path=os.path.join(args.bop_parent_path, 'tless'),
    model_type='cad', mm2m=True
)

# load BOP dataset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path=os.path.join(args.bop_parent_path, 'tless'))

# set shading and hide objects
for obj in target_bop_objs:
    obj.set_shading_mode('auto')
    obj.hide(True)

# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)

# sample light color and strength from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(100)

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

# Sample bop objects for a single scene
sampled_target_bop_objs = list(np.random.choice(target_bop_objs, size=20, replace=False))
    
# Randomize materials and set physics
for obj in sampled_target_bop_objs:
    ind_rng = np.random.default_rng()
    random_cc_texture = ind_rng.choice(cc_textures)

    obj.replace_materials(random_cc_texture)
    if not obj.has_uv_mapping():
        obj.add_uv_mapping("smart")
    obj.enable_rigidbody(True, mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)
    obj.hide(False)

# Sample two light sources
light_plane_material.make_emissive(emission_strength=np.random.uniform(3, 6),
                                   emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))
light_plane.replace_materials(light_plane_material)
light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
location = bproc.sampler.shell(center=[0, 0, 0], radius_min=1, radius_max=1.5,
                               elevation_min=5, elevation_max=89)
light_point.set_location(location)

# sample CC Texture and assign to room planes
random_room_texture = np.random.choice(cc_textures)
for plane in room_planes:
    plane.replace_materials(random_room_texture)

# Sample object poses and check collisions
bproc.object.sample_poses(objects_to_sample=sampled_target_bop_objs,
                          sample_pose_func=sample_pose_func,
                          max_tries=1000)

# Physics Positioning
bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                  max_simulation_time=10,
                                                  check_object_interval=1,
                                                  substeps_per_frame=20,
                                                  solver_iters=25)

# BVH tree used for camera obstacle checks
bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs)

cam_poses = 0
while cam_poses < args.views:
    location = bproc.sampler.shell(center=[0, 0, 0],
                                   radius_min=0.65,
                                   radius_max=0.94,
                                   elevation_min=5,
                                   elevation_max=89)
    poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=15, replace=False))
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location,
                                                              inplane_rot=np.random.uniform(-3.14159, 3.14159))
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
        bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
        cam_poses += 1

# render the whole pipeline
data = bproc.renderer.render()

# Write data in bop format
bproc.writer.write_bop(os.path.join(args.output_dir, f'bop_data_{args.variant}'),
                       target_objects=sampled_target_bop_objs,
                       dataset=f'tless_random_texture_{args.variant}',
                       depth_scale=0.1,
                       depths=data["depth"],
                       colors=data["colors"],
                       color_file_format="JPEG",
                       ignore_dist_thres=10,
                       append_to_existing_output=True)

for obj in sampled_target_bop_objs:
    obj.disable_rigidbody()
    obj.hide(True)
