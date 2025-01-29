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

class TransparentMaterials: #CHANGE
    def __init__(self, blend_file_path="TransparentMaterials.blend"):
        self.materials = {}

        objs = bproc.loader.load_blend(blend_file_path)

        for obj in objs:
            material = obj.get_materials()[0]
            name = obj.get_name()
            if name in self.materials:
                raise ValueError(f"Material with name {name} already exists.")
            self.materials[name] = material
        
        print(f"Successfully loaded {len(self.materials)} materials")

        bproc.object.delete_multiple(objs)
        print(self.materials)

    def get_random_material(self):
        return np.random.choice(list(self.materials.values()))

def set_material_properties(obj, cc_textures, randomize=True, transparent_materials=None):
        if randomize:
            random_cc_texture = np.random.choice(cc_textures)
            obj.replace_materials(random_cc_texture)

        if not obj.has_uv_mapping():
            obj.add_uv_mapping("smart")

        mat = obj.get_materials()[0]

        if obj.get_cp("obj_name") == "glass":
            obj.replace_materials(transparent_materials.get_random_material())  
        
        elif obj.get_cp("obj_name") == "cutlery":
            mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
            mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0))
            mat.set_principled_shader_value("Metallic", np.random.uniform(0.3, 1.0))
            mat.set_principled_shader_value("Alpha", 1.0)
            grey_col = np.random.uniform(0.1, 0.7)   
            mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])   
        
        else:
            mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
            mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0)) 

        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.hide(False)
        return obj

def load_meshes(mesh_dir):
    target_objs = []
    # Get the sorted list of subfolders in the mesh_dir
    category_folders = sorted([d for d in os.listdir(mesh_dir) if os.path.isdir(os.path.join(mesh_dir, d))])
    i = 0
    
    for idx, category_folder in enumerate(category_folders):
        category_path = os.path.join(mesh_dir, category_folder)
        
        for model_file in os.listdir(category_path):
            if model_file.endswith('.obj'):
                full_path = os.path.join(category_path, model_file)
                obj = bproc.loader.load_obj(full_path)[0]

                print(i)
                print(model_file)
                print()
                obj.set_cp("category_id", i)
                obj.set_cp("cat_id", idx)
                obj_name = model_file.split("-")[0]
                obj.set_cp("obj_name", model_file.split("-")[0])
                
                obj.set_name(model_file)
                
                obj.hide(True)
                obj.set_shading_mode('auto')
                materials = obj.get_materials()

                if not materials:  # If the materials list is empty
                    print(f"No materials found for {model_file}. Assigning a new material.")
                    
                    # Create a new material
                    new_material = bproc.material.create("default_material")
                    
                    # Set properties for the material (optional, adjust as needed)
                    new_material.set_principled_shader_value("Base Color", [1.0, 1.0, 1.0, 1.0])  # White color
                    new_material.set_principled_shader_value("Roughness", 0.5)  # Default roughness
                    
                    # Assign the new material to the object
                    obj.replace_materials(new_material)
                else:
                    print(materials[0])  # Print the first material if it exists

                bbox = obj.get_bound_box()
                min_coords = np.min(bbox, axis=0)
                max_coords = np.max(bbox, axis=0)
                diagonal = np.linalg.norm(max_coords - min_coords)
                scaling_factor = (1.0 / diagonal) * 0.1
                scaling_factor = 1.0

                obj.set_scale([scaling_factor,scaling_factor,scaling_factor])

                # Apply scale and ensure quads are converted to tris
                bpy.context.view_layer.objects.active = obj.blender_obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
                bpy.ops.object.mode_set(mode='OBJECT')               
                
                target_objs.append(obj)
                i = i + 1
    
    return target_objs
   
def sample_pose_func(obj: bproc.types.MeshObject):
        min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
        max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
        obj.set_location(np.random.uniform(min, max))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())

def render(config):

    bproc.init()
    mesh_dir = os.path.join(dirname, config["models_dir"])

    target_objs = load_meshes(mesh_dir)

    transparent_materials = TransparentMaterials()

    dataset_name = config["dataset_name"]

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

    # load cc_textures
    cc_textures = bproc.loader.load_ccmaterials(config['texture_dir'])
       
    # activate depth rendering without antialiasing and set amount of samples for color rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(50)

    bproc.renderer.set_light_bounces( #CHANGE
        glossy_bounces=32, 
        max_bounces=32, 
        transmission_bounces=32, 
        transparent_max_bounces=50, 
        volume_bounces=32)  

    bproc.camera.set_intrinsics_from_K_matrix(np.reshape(config["cam"]["K"], (3, 3)), 
                                                config["cam"]["width"], 
                                                config["cam"]["height"])

    def sample_initial_pose(obj: bproc.types.MeshObject):
        obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                    min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
        #obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, 0]))
        #obj.set_rotation_euler([np.pi/2, 0, np.pi/2])
        initial_rotation = [np.pi/2, 0, np.pi/2]
        obj.set_rotation_euler(initial_rotation)
        
        #obj.set_rotation_euler(np.random.uniform([np.pi/2, 0, np.pi/2], [np.pi/2, 2*np.pi, np.pi/2]))
        #obj.set_rotation_euler([0,0,0])
        # rotation_matrix = np.array([
        #     [0, 0, 1],
        #     [1, 0, 0],
        #     [0, 1, 0]
        # ])
        # obj.set_rotation_mat(rotation_matrix)
        # Define a function that samples 6-DoF poses
            
        # Step 2: Get the current rotation matrix after applying initial rotation
        current_rotation_matrix = obj.get_rotation_mat()  # This is a 3x3 rotation matrix
        
        # Step 3: Generate a random y-axis rotation (around the new y-axis)
        random_y_rotation = np.random.uniform(0, 2 * np.pi)
        
        # Step 4: Build a random rotation matrix around the y-axis
        # Using the rotation matrix for a rotation around the y-axis
        rotation_matrix_y = np.array([
            [np.cos(random_y_rotation), 0, np.sin(random_y_rotation)],
            [0, 1, 0],
            [-np.sin(random_y_rotation), 0, np.cos(random_y_rotation)]
        ])
        
        # Step 5: Combine the current rotation matrix with the random rotation matrix
        new_rotation_matrix = current_rotation_matrix @ rotation_matrix_y
        
        # Step 6: Apply the new combined rotation to the object
        obj.set_rotation_mat(new_rotation_matrix)
    
    for i in range(config["num_scenes"]):

        # Sample bop objects for a scene
        sampled_target_objs = list(np.random.choice(target_objs, size=config['num_objects'], replace=False))

        # Randomize materials and set physics
        for obj in (sampled_target_objs):
            obj = set_material_properties(obj, cc_textures, randomize=False, transparent_materials=transparent_materials)

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
       

        bproc.object.sample_poses(objects_to_sample = sampled_target_objs,
                                    sample_pose_func = sample_pose_func, 
                                    max_tries = 100)
    
        if i % 2 == 0:   
            # def sample_initial_pose(obj):
            #     obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
            #                                                 min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
            #     initial_rotation = [np.pi/2, 0, np.pi/2]
            #     obj.set_rotation_euler(initial_rotation)
                
            #     current_rotation_matrix = obj.get_rotation_mat()
                
            #     random_y_rotation = np.random.uniform(0, 2 * np.pi)
            #     rotation_matrix_y = np.array([
            #         [np.cos(random_y_rotation), 0, np.sin(random_y_rotation)],
            #         [0, 1, 0],
            #         [-np.sin(random_y_rotation), 0, np.cos(random_y_rotation)]
            #     ])
                
            #     # Step 5: Combine the current rotation matrix with the random rotation matrix
            #     new_rotation_matrix = current_rotation_matrix @ rotation_matrix_y
                
            #     # Step 6: Apply the new combined rotation to the object
            #     obj.set_rotation_mat(new_rotation_matrix)

            bproc.object.sample_poses_on_surface(objects_to_sample=sampled_target_objs,
                                                surface=room_planes[0],
                                                sample_pose_func=sample_initial_pose,
                                                min_distance=0.01,
                                                max_distance=0.2)

        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                        max_simulation_time=10,
                                        check_object_interval=1,
                                        substeps_per_frame = 20,
                                        solver_iters=25)

        # BVH tree used for camera obstacle checks
        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_objs)
        
        cam_poses = 0
        while cam_poses < config["img_per_scene"]:
            # Sample location
            location = bproc.sampler.shell(center = [0, 0, 0],
                                    radius_min = 0.25,
                                    radius_max = 0.6,
                                    elevation_min = 5,
                                    elevation_max = 89)
            # Determine point of interest in scene as the object closest to the mean of a subset of objects
            poi = bproc.object.compute_poi(np.random.choice(sampled_target_objs, size=max(5, len(sampled_target_objs)-1), replace=False)) #
            # Compute rotation based on vector going from location towards poi
            # rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-np.pi/2.0, np.pi/2.0))
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))

            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            
            # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
                # Persist camera pose
                bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
                cam_poses += 1

        # render the whole pipeline
        data = bproc.renderer.render()

        for plane in room_planes:
            plane.hide(True)
        
        data.update(bproc.renderer.render_nocs())

        for plane in room_planes:
            plane.hide(False)
            
        # Write data in bop format
        bproc.writer.write_bop_with_nocs(os.path.join(config["output_dir"], 'bop_data'),
                            target_objects = sampled_target_objs,
                            dataset = dataset_name,
                            depth_scale = 0.1,
                            depths = data["depth"],
                            colors = data["colors"],
                            nocs = data["nocs"],
                            color_file_format = "PNG",
                            ignore_dist_thres = 10,
                            calc_mask_info_coco = True,
                            )

        for obj in sampled_target_objs:
            obj.disable_rigidbody()
            obj.hide(True)

if __name__ == "__main__":
    config_path = "./housecat6d_cfg.yaml"

    dirname = os.path.dirname(__file__) #TODO

    with open(os.path.join(dirname, config_path), "r") as stream:
        config = yaml.safe_load(stream)
    render(config)


    
    

    