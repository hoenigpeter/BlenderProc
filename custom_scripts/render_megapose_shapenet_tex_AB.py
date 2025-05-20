import blenderproc as bproc
# from curses.panel import bottom_panel
# from logging import raiseExceptions
# from sre_parse import CATEGORIES
# from unicodedata import category
import argparse
import os
import numpy as np
import yaml
import sys
import imageio
import random
import shutil
import bpy
import glob
import json
from tqdm import tqdm
import re
import cv2
from mathutils import Matrix, Vector
from collections import defaultdict

from scipy.spatial.transform import Rotation
import webdataset as wds
import tarfile

def set_material_properties(obj, cc_textures, randomize=True):
        if not obj.has_uv_mapping():
            obj.add_uv_mapping("smart")

        if randomize:
            random_cc_texture = np.random.choice(cc_textures)
            obj.replace_materials(random_cc_texture)

        mat = obj.get_materials()[0]    
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Metallic", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Alpha", 1.0)
        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.hide(False)

def convert_to_blender_TWC(TWC):
    flip = np.eye(4)
    flip[1,1] = -1
    flip[2,2] = -1
    return TWC @ flip

def extract_synset_model(label):
    match = re.match(r"shapenet_(\w+)_(\w+)", label)
    if match:
        synsetId, modelId = match.groups()
        return synsetId, modelId
    else:
        return None, None

def compute_transform(R, t):
    R_matrix_scipy = Rotation.from_quat(R).as_matrix()
    transform_matrix = np.eye(4)

    transform_matrix[:3, :3] = R_matrix_scipy
    transform_matrix[:3, 3] = t

    return transform_matrix

def render(output_directory, num, cc_textures):

    shapenet_directory = "/ssd3/shapenetcorev2/models_orig"
    megapose_path = "/hdd/megapose_shapenet" 

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Directory '{output_directory}' created.")
    else:
        print(f"Directory '{output_directory}' already exists.")

    folder_num = f"{num:08d}"
    folder_path = os.path.join(megapose_path, folder_num)
    folder_path = megapose_path + "/" + folder_num + ".tar"
    print(f"Folder path: {folder_path}")

    # Reset dataset iterator
    dataset = (
        wds.WebDataset(folder_path)
        .decode("pil")
        .to_tuple("__key__", "camera_data.json", "object_datas.json", "infos.json")
    )

    current_scene_id = None
    scene_objs = []
    instance_ids = []
    infos_list = []
    camera_poses = []
    object_data_list = []
    camera_data_list = []

    # create room
    room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
                bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

    # sample light color and strenght from ceiling
    light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')

    # sample point light on shell
    light_point = bproc.types.Light()
    light_point.set_energy(100)

    shard_pattern = os.path.join(output_directory, "shard-%06d.tar")
    shard_size = 1000
    
    with wds.ShardWriter(shard_pattern, maxcount=shard_size) as writer:            
        for instance_id, camera_data, object_data, infos in tqdm(dataset, desc="Rendering"):
            scene_id = int(infos["scene_id"])
            view_id = int(infos["view_id"])
            print("############## view id: ", view_id)
            instance_ids.append(instance_id)
            infos_list.append(infos)
            object_data_list.append(object_data)
            camera_data_list.append(camera_data)

            # New scene: reset objects
            if current_scene_id != scene_id:
                current_scene_id = scene_id
                for obj in scene_objs:
                    obj.delete()
                scene_objs = []

            if view_id == 0:
                # sample CC Texture and assign to room planes
                random_cc_texture = np.random.choice(cc_textures)
                for plane in room_planes:
                    plane.replace_materials(random_cc_texture)

                # Sample two light sources
                light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                                emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
                light_plane.replace_materials(light_plane_material)
                light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
                location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                                        elevation_min = 5, elevation_max = 89)
                light_point.set_location(location)

                for idx, obj_entry in enumerate(object_data):

                    synsetId, modelId = extract_synset_model(obj_entry["label"])

                    mesh_path = os.path.join(shapenet_directory, synsetId, modelId, "models/model_normalized.obj")
                    if not os.path.exists(mesh_path):
                        print(f"[SKIP] Missing model: {mesh_path}")
                        continue

                    base_obj = bproc.loader.load_obj(mesh_path)[0]
                    base_obj.set_scale([0.1, 0.1, 0.1])

                    TWO = obj_entry["TWO"]
                    obj_pose = compute_transform(TWO[0], TWO[1])

                    base_obj.set_location(obj_pose[:3, 3])
                    base_obj.set_rotation_mat(obj_pose[:3, :3])
                    base_obj.set_cp("obj_id", idx)

                    #set_material_properties(base_obj, cc_textures, randomize=True)

                    scene_objs.append(base_obj)

            cam_K = np.array(camera_data["K"]).reshape(3, 3)
            TWC = compute_transform(camera_data["TWC"][0], camera_data["TWC"][1])
            TWC_blender = convert_to_blender_TWC(TWC)

            bproc.camera.set_intrinsics_from_K_matrix(
                cam_K,
                camera_data["resolution"][1],
                camera_data["resolution"][0]
            )

            camera_poses.append(TWC_blender)

            if view_id == 19:
                print(f"[Render] Scene {scene_id}")

                for idx, camera_pose in enumerate(camera_poses):
                    bproc.camera.add_camera_pose(camera_pose, frame=idx)

                np.random.seed()
                for obj in scene_objs:
                    set_material_properties(obj, cc_textures, randomize=True)

                data_A = bproc.renderer.render()
                print("Rendering A completed")

                # for idx, camera_pose in enumerate(camera_poses):
                #     bproc.camera.add_camera_pose(camera_pose, frame=idx)

                np.random.seed()
                for obj in scene_objs:
                    set_material_properties(obj, cc_textures, randomize=True)

                data_B = bproc.renderer.render()
                print("Rendering B completed")

                for idx, _ in enumerate(instance_ids):
                    rgb_data_A = data_A['colors'][idx][..., :3]
                    print("rgb_data_A.shape: ", rgb_data_A.shape)
                    rgb_data_B = data_B['colors'][idx][..., :3]
                    print("rgb_data_B.shape: ", rgb_data_B.shape)
                
                    output_data = {
                        "__key__": instance_ids[idx],
                        "rgb_A.png": rgb_data_A,
                        "rgb_B.png": rgb_data_B,
                        "camera_data.json": camera_data_list[idx],
                        "object_datas.json": object_data_list[idx],
                        "infos.json": infos_list[idx],
                    }
                    writer.write(output_data)

                    print("saved:", instance_ids[idx])

                # Clean up for next scene
                for obj in scene_objs:
                    obj.hide(True)
                    obj.delete()

                scene_objs = []
                camera_poses = []
                instance_ids = []

if __name__ == "__main__":
    
    bproc.init()
    texture_dir = "/home/hoenig/BlenderProc/resources/cc_textures"
    cc_textures = bproc.loader.load_ccmaterials(texture_dir)
    output_dir = "./test"
    
    for num in range(1030):
        render(output_dir, num, cc_textures)
        bproc.clean_up()