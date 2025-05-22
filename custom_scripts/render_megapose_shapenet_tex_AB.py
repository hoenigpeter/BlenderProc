import blenderproc as bproc
import os
import numpy as np
import sys
import bpy
from tqdm import tqdm
import re

from scipy.spatial.transform import Rotation
import webdataset as wds

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

def main():
    # Parse arguments
    num = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    batch_iter = int(sys.argv[3])

    bproc.init()

    print(f"[INFO] Running num={num}, batch_size={batch_size}, batch_iter={batch_iter}")

    texture_dir = "/home/hoenig/BlenderProc/resources/cc_textures"
    output_directory = "/ssd3/megapose_tex_AB"

    shapenet_directory = "/ssd3/shapenetcorev2/models_bop-renderer_scale=0.1"
    megapose_path = "/ssd3/megapose_shapenet_stripped" 

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Directory '{output_directory}' created.")
    else:
        print(f"Directory '{output_directory}' already exists.")

    source_tar_num = f"{num:08d}"
    source_folder_path = os.path.join(megapose_path, source_tar_num)
    source_folder_path = megapose_path + "/" + source_tar_num + ".tar"
    print(f"Folder path: {source_folder_path}")

    # Reset dataset iterator
    dataset = (
        wds.WebDataset(source_folder_path)
        .decode("pil")
        .to_tuple("__key__", "camera_data.json", "object_datas.json", "infos.json")
    )

    dataset_list = list(dataset)
    total_samples = len(dataset_list)

    samples_per_batch = total_samples // batch_size
    start_idx = batch_iter * samples_per_batch
    end_idx = (batch_iter + 1) * samples_per_batch if batch_iter < batch_size - 1 else total_samples
    print("start_idx: ", start_idx)
    print("end_idx: ", end_idx)
    batch = dataset_list[start_idx:end_idx]

    current_scene_id = None
    scene_objs = []
    instance_ids = []
    infos_list = []
    camera_poses = []
    object_data_list = []
    camera_data_list = []
    obj_names = []

    output_tar_path = os.path.join(output_directory, f"{num:08d}_{batch_iter:02d}.tar")

    cc_textures = bproc.loader.load_ccmaterials(texture_dir)
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
    light_point.set_energy(100)

    bproc.renderer.set_max_amount_of_samples(50)

    with wds.TarWriter(output_tar_path) as writer:            
        for instance_id, camera_data, object_data, infos in tqdm(batch, desc="Rendering"):
            scene_id = int(infos["scene_id"])
            view_id = int(infos["view_id"])
            print("############## view id: ", view_id)

            instance_ids.append(instance_id)
            infos_list.append(infos)
            object_data_list.append(object_data)
            camera_data_list.append(camera_data)

            if view_id == 0:

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

                scene_objs = []
                obj_names = []
                for idx, obj_entry in enumerate(object_data):

                    synsetId, modelId = extract_synset_model(obj_entry["label"])

                    mesh_path = os.path.join(shapenet_directory, synsetId, modelId, "models/model_normalized_scaled.ply")

                    base_obj = bproc.loader.load_obj(mesh_path)[0]
                    base_obj.set_scale([0.001, 0.001, 0.001])

                    # Get the bounding box corners (8 corners)
                    bbox = base_obj.get_bound_box()

                    # Compute size: axis-aligned bounding box (AABB)
                    bbox = np.array(bbox)
                    min_corner = bbox.min(axis=0)
                    max_corner = bbox.max(axis=0)
                    bbox_size = max_corner - min_corner

                    print("Bounding box size (X, Y, Z):", bbox_size)

                    TWO = obj_entry["TWO"]
                    obj_pose = compute_transform(TWO[0], TWO[1])

                    base_obj.set_location(obj_pose[:3, 3])
                    base_obj.set_rotation_mat(obj_pose[:3, :3])
                    base_obj.set_cp("obj_id", idx)

                    obj_name = obj_entry['label']
                    base_obj.set_name(obj_name)

                    scene_objs.append(base_obj)
                    obj_names.append(obj_name)

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

                print("Add camera poses")
                for idx, camera_pose in enumerate(camera_poses):
                    bproc.camera.add_camera_pose(camera_pose, frame=idx)

                print("Texture Scene A")
                print("Preparing scene A...")
                np.random.seed(np.random.randint(0, 1_000_000))
                print("num of scene_objs: ", len(scene_objs))
                for idx, obj in enumerate(scene_objs):
                    print(f"{idx} Obj: {obj.get_name()}, has UV: {obj.has_uv_mapping()}")
                    set_material_properties(obj, cc_textures, randomize=True)

                print("Starting render...")
                try:
                    data_A = bproc.renderer.render()
                except Exception as e:
                    print(f"[ERROR] Render Scene A failed: {e}")
                    return
                print("Rendering Scene A completed")

                print("Texture Scene B")
                print("Preparing scene B...")
                np.random.seed(np.random.randint(0, 1_000_000))
                for idx, obj in enumerate(scene_objs):
                    print(f"{idx} Obj: {obj.get_name()}, has UV: {obj.has_uv_mapping()}")
                    set_material_properties(obj, cc_textures, randomize=True)

                print("Starting render...")
                try:
                    data_B = bproc.renderer.render()
                except Exception as e:
                        print(f"[ERROR] Render Scene A failed: {e}")
                        return
                print("Rendering Scene B completed")

                for idx, _ in enumerate(instance_ids):
                    rgb_data_A = data_A['colors'][idx][..., :3]
                    rgb_data_B = data_B['colors'][idx][..., :3]

                    print(instance_ids[idx])
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
                camera_data_list = []
                object_data_list = []
                infos_list = []
                obj_names = []   

if __name__ == "__main__":
    main()