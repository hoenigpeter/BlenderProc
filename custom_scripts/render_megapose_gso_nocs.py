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

def convert_to_blender_TWC(TWC):
    # Flip coordinate system: rotate 180° around X to match Blender's camera look direction
    flip = np.eye(4)
    flip[1,1] = -1
    flip[2,2] = -1
    return TWC @ flip

def load_filenames(folder_path):
    filenames = defaultdict(list)

    for filename in sorted(os.listdir(folder_path)):
        instance_id, modality = filename.split(".", 1)
        modality_name, _ = modality.split(".", 1)

        filenames[instance_id].append((modality_name, filename))

    return filenames

def load_json_data(folder_path, instance_id, file_extension=".object_datas.json"):
    json_data = None
    json_filename = None

    for filename in os.listdir(folder_path):
        if filename.startswith(instance_id) and filename.endswith(file_extension):
            json_filename = filename
            break

    if json_filename:
        json_file_path = os.path.join(folder_path, json_filename)
        with open(json_file_path, "r") as json_file:
            json_data = json.load(json_file)
    else:
        raise FileNotFoundError(f"No JSON file found for instance ID {instance_id}")

    return json_data

def compute_transform(R, t):
    # Convert quaternion to rotation matrix
    R_matrix_scipy = Rotation.from_quat(R).as_matrix()
    transform_matrix = np.eye(4)

    transform_matrix[:3, :3] = R_matrix_scipy
    transform_matrix[:3, 3] = t

    return transform_matrix

def render(output_directory, num):

    gso_directory = "/ssd3/google_scanned_objects/models_normalized"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Directory '{output_directory}' created.")
    else:
        print(f"Directory '{output_directory}' already exists.")

    output_directory = os.path.join(output_directory, f"{num:08d}")

    megapose_path = "/hdd/megapose_gso" 
    folder_num = f"{num:08d}"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Directory '{output_directory}' created.")
    else:
        print(f"Directory '{output_directory}' already exists.")

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

    for instance_id, camera_data, object_data, infos in tqdm(dataset, desc="Rendering"):
        scene_id = int(infos["scene_id"])
        view_id = int(infos["view_id"])
        print("############## view id: ", view_id)
        instance_ids.append(instance_id)

        # New scene: reset objects
        if current_scene_id != scene_id:
            current_scene_id = scene_id
            for obj in scene_objs:
                obj.delete()
            scene_objs = []

        if view_id == 0:
            for idx, obj_entry in enumerate(object_data):

                model_label = obj_entry["label"]
                print("model_label: ", model_label)

                mesh_path = os.path.join(gso_directory, model_label[4:],  "meshes/model.obj")
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
                scene_objs.append(base_obj)

        cam_K = np.array(camera_data["K"]).reshape(3, 3)
        TWC = compute_transform(camera_data["TWC"][0], camera_data["TWC"][1])
        TWC_blender = convert_to_blender_TWC(TWC)

        bproc.camera.set_intrinsics_from_K_matrix(
            cam_K,
            camera_data["resolution"][1],
            camera_data["resolution"][0]
        )
        bproc.camera.add_camera_pose(TWC_blender, frame=view_id)

        if view_id == 39:
            print(f"[Render] Scene {scene_id}")
            data = bproc.renderer.render_nocs()
            for render_id, inst_id in enumerate(instance_ids):
                nocs_rgb = data['nocs'][render_id][..., :3]
                nocs_bgr = (np.clip(nocs_rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1]
                output_path = os.path.join(output_directory, f"{inst_id}.nocs.png")
                cv2.imwrite(output_path, nocs_bgr)

            # Clean up for next scene
            for obj in scene_objs:
                obj.hide(True)
                obj.delete()

            scene_objs = []
            instance_ids = []

            bproc.clean_up()

if __name__ == "__main__":
    
    bproc.init()
    for num in range(951, 1050):
        output_dir = "./megapose_nocs_gso"
        render(output_dir, num)
