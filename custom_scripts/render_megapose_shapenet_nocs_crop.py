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
import imageio
import io
import PIL
from PIL import Image
from scipy.spatial.transform import Rotation
import webdataset as wds
import tarfile
import matplotlib.pyplot as plt
import pyfastnoisesimd as fns

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../temp/nocs_renderer'))
sys.path.append(parent_dir)

from utils import augment_depth, backproject, estimate_pointnormals

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

def extract_synset_model(label):
    match = re.match(r"shapenet_(\w+)_(\w+)", label)
    if match:
        synsetId, modelId = match.groups()
        return synsetId, modelId
    else:
        return None, None

def get_name_from_taxonomy(synset_data, synset_id):
    for item in synset_data:
        if item["synsetId"] == synset_id:
            return item["name"]
    return None

def compute_transform(R, t):
    # Convert quaternion to rotation matrix
    R_matrix_scipy = Rotation.from_quat(R).as_matrix()
    transform_matrix = np.eye(4)

    transform_matrix[:3, :3] = R_matrix_scipy
    transform_matrix[:3, 3] = t

    return transform_matrix

def load_depth_image(depth_bytes, depth_scale_factor= 1.0):
    depth_image = imageio.imread(depth_bytes).astype(np.float32)
    depth_image /= depth_scale_factor
    return depth_image

def load_image(data):
    with io.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.convert("rgb".upper())
        img = img.convert("RGB")
        return img

def extract_binary_masks(segmentation_image):
    category_ids = np.unique(segmentation_image)
    binary_masks = [(segmentation_image == cat_id).astype(np.uint8)
                    for cat_id in category_ids if cat_id != 0]
    return binary_masks
        
def render(output_directory, num):

    shapenet_directory = "/ssd3/shapenetcorev2/models_orig"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Directory '{output_directory}' created.")
    else:
        print(f"Directory '{output_directory}' already exists.")

    output_directory = os.path.join(output_directory, f"{num:08d}")

    megapose_path = "/hdd/megapose_shapenet" 
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
    # dataset = (
    #     wds.WebDataset(folder_path)
    #     .decode("pil")
    #     .to_tuple("__key__", "rgb.png", "depth.png", "segmentation.png", "camera_data.json", "object_datas.json", "infos.json")
    # )

    dataset = (wds.WebDataset(folder_path) \
            .decode() \
            .to_tuple("__key__", "rgb.png", "depth.png", "segmentation.png", "camera_data.json", "object_datas.json", "infos.json") \
            .map_tuple(
                lambda __key__: __key__,
                lambda rgb: load_image(rgb),
                lambda depth: load_depth_image(depth, depth_scale_factor=1000.0),
                lambda segmentation: load_depth_image(segmentation),
                lambda camera_data: camera_data,
                lambda object_datas: object_datas,
                lambda infos: infos,
            ))
    
    current_scene_id = None
    scene_objs = []
    instance_ids = []

    rgb_crops = []
    mask_crops = []
    nocs_crops = []
    normal_crops = []

    # iterate through image after image
    for instance_id, rgb_image, depth_image, segmentation_image, camera_data, object_data, infos in tqdm(dataset, desc="Rendering"):
        scene_id = int(infos["scene_id"])
        view_id = int(infos["view_id"])
        segmentation_image = segmentation_image.astype(np.uint16)
        print("############## view id: ", view_id)
        instance_ids.append(instance_id)

        masks = extract_binary_masks(segmentation_image)
        #save_binary_masks(segmentation_image, "output/masks")

        depth_aug = augment_depth(depth_image)
        depth_aug_norm = (depth_aug - depth_aug.min()) / (depth_aug.max() - depth_aug.min()) * 255
        depth_aug_norm = depth_aug_norm.astype(np.uint8)
        #cv2.imwrite(os.path.join(output_dir, f"{img_id}_depth_noisy.png"), depth_aug_norm)

        pointcloud, _ = backproject(depth_aug, fx, fy, cx, cy, instance_mask=None)
        normals = estimate_pointnormals(pointcloud, fx, fy, cx, cy, height, width, orient_normals=True)
        normals_uint8_with_aug = (normals * 255).astype(np.uint8)


        # New scene: reset objects
        if current_scene_id != scene_id:
            current_scene_id = scene_id
            for obj in scene_objs:
                obj.delete()
            scene_objs = []

        if view_id == 0:
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
                scene_objs.append(base_obj)

        cam_K = np.array(camera_data["K"]).reshape(3, 3)
        TWC = compute_transform(camera_data["TWC"][0], camera_data["TWC"][1])
        TWC_blender = convert_to_blender_TWC(TWC)

        bproc.camera.set_intrinsics_from_K_matrix(
            cam_K,
            camera_data["resolution"][1],
            camera_data["resolution"][0]
        )
        bproc.camera.add_camera_pose(TWC_blender)

        print(f"[Render] Scene {scene_id}")
        data = bproc.renderer.render_nocs()
        nocs_rgb = data['nocs'][0][..., :3]
        
        for idx, obj_entry in enumerate(object_data):
            unique_id = obj_entry["unique_id"]
            visib_frac = obj_entry["visib_fract"]
            mask = masks[idx]


            print("idx: ", idx)
            print("unique_id: ", unique_id)
            print("visib_fract: ", visib_frac)
        

        # nocs_rgb = data['nocs'][0][..., :3]
        # nocs_bgr = (np.clip(nocs_rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1]
        # output_path = os.path.join(output_directory, f"{inst_id}.nocs.png")
        # cv2.imwrite(output_path, nocs_bgr)

        if view_id == 19:
            # Clean up for next scene
            for obj in scene_objs:
                obj.hide(True)
                obj.delete()
            scene_objs = []

        #     instance_ids = []

        #     bproc.clean_up()

        break

def add_nocs_images_to_tar(tar_path, nocs_dir):
    """
    Adds all .nocs.png files from nocs_dir into the tar archive at tar_path.
    """
    with tarfile.open(tar_path, "a") as tar:
        for file_name in os.listdir(nocs_dir):
            if file_name.endswith(".nocs.png"):
                full_path = os.path.join(nocs_dir, file_name)
                tar_name = file_name  # or f"{file_name}" if you want a subfolder inside the tar
                tar.add(full_path, arcname=tar_name)
                print(f"Added {full_path} to {tar_path} as {tar_name}")

if __name__ == "__main__":
    
    bproc.init()
    for num in range(1):
        output_dir = "/ssd3/megapose_nocs"

        dirname = os.path.dirname(__file__)
        render(output_dir, num)
        # add_nocs_images_to_tar(f"/hdd/megapose_shapenet/{num:08d}.tar", output_dir + "/" + f"{num:08d}")
        # shutil.rmtree(output_dir)
        # print(f"Temporary directory '{output_dir}' removed.")