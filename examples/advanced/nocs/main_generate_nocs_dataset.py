import blenderproc as bproc
import argparse
import os 
import trimesh 
import numpy as np
from tqdm import tqdm

def calc_cam_poses(points):
    camera_poses = []

    for point in points:
        center = np.array([0.0, 0.0, 0.0])
        direction = center - point
        direction /= np.linalg.norm(direction)

        z_axis = np.array([0, 0, -1])
        rotation_axis = np.cross(z_axis, direction)
        rotation_angle = np.arccos(np.dot(z_axis, direction))
        rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)[:3, :3]

        # Calculate translation vector
        translation_vector = point

        # Set camera pose (rotation)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rotation_matrix
        camera_pose[:3, 3] = translation_vector

        camera_poses.append(rotation_matrix)

    return camera_poses

bproc.init()

output_dir = "/hdd2/nocs_category_level_v2/"
shapenet_directory = "/hdd2/real_camera_dataset/obj_models/train"
#shapenet_directory = "/hdd/datasets_bop/shapenetcorev2/models_orig/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

synset_ids = sorted(os.listdir(shapenet_directory))
synset_id = "03642806"

synset_dir = output_dir + "/" + synset_id

if not os.path.exists(synset_dir):
    os.makedirs(synset_dir)

outputs = ["colors", "normals", "nocs", "instance_segmaps"]

for output in outputs:
    subdir_path = os.path.join(synset_dir, output)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)

source_ids = sorted(os.listdir(shapenet_directory + "/" + synset_id))

bproc.camera.set_resolution(128, 128)
# sample light color and strenght from ceiling
# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(100)

light_point_2 = bproc.types.Light()
light_point_2.set_energy(100)

for idx, source_id in tqdm(enumerate(source_ids), total=len(source_ids), desc="Processing Objects"):
    bproc.utility.reset_keyframes()
    #shapenet_obj = bproc.loader.load_shapenet(shapenet_directory, used_synset_id=synset_id, used_source_id=source_id, move_object_origin=False)
    shapenet_obj = bproc.loader.load_shapenet(shapenet_directory, used_synset_id=synset_id, used_source_id=source_id, move_object_origin=False)

    points = trimesh.creation.icosphere(subdivisions=2, radius=2).vertices
    camera_poses = calc_cam_poses(points)

    # Sample two light sources

    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    light_point.set_location(location)

    light_point_2.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    light_point_2.set_location(location)

    for i, camera_pose in enumerate(camera_poses):
        # Sample random camera location around the object
        #location = bproc.sampler.sphere([0, 0, 0], radius=2, mode="SURFACE")
        location = points[i]
        # Compute rotation based on vector going from location towards the location of the ShapeNet object
        rotation_matrix = bproc.camera.rotation_from_forward_vec(shapenet_obj.get_location() - location)
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, camera_pose)
        bproc.camera.add_camera_pose(cam2world_matrix)

    bproc.renderer.enable_normals_output()
    #bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by=["instance"])
    data = bproc.renderer.render()

    # Render NOCS
    data.update(bproc.renderer.render_nocs())
    bproc.writer.write_images(synset_dir, idx, data)
    shapenet_obj.hide(True)