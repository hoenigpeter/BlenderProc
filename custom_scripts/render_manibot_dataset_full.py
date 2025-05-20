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

def set_material_properties(obj, cc_textures, randomize=True):
        if randomize:
            random_cc_texture = np.random.choice(cc_textures)
            obj.replace_materials(random_cc_texture)

        mat = obj.get_materials()[0]    
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Metallic", np.random.uniform(0, 1.0))
        #mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Alpha", 1.0)
        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.hide(False)
        return obj
  
def sample_pose_func(obj: bproc.types.MeshObject):
        min = np.random.uniform([-0.3, -0.3, 0.0], [-0.3, -0.3, 0.0])
        max = np.random.uniform([0.3, 0.3, 0.4], [0.3, 0.3, 0.6])
        obj.set_location(np.random.uniform(min, max))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())

def render(config):

    bproc.init()
    bop_parent_path = "/ssd3/datasets_bop"

    target_objs = bproc.loader.load_bop_objs(
        bop_dataset_path=os.path.join(bop_parent_path, 'manibot'),
        model_type="cad", object_model_unit='m'
    )

    dataset_name = config["dataset_name"]

    # create room
    room_size = max(config["cam"]["radius_max"] * 1.1, 6)
    room_planes = [bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1]),
                bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1], location=[0, -room_size, room_size], rotation=[-1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1], location=[0, room_size, room_size], rotation=[1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1], location=[room_size, 0, room_size], rotation=[0, -1.570796, 0]),
                bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1], location=[-room_size, 0, room_size], rotation=[0, 1.570796, 0])]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

    # sample light color and strenght from ceiling
    light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')

    # sample point light on shell
    light_point = bproc.types.Light()


    # load cc_textures
    cc_textures = bproc.loader.load_ccmaterials(config['texture_dir'])
       
    # activate depth rendering without antialiasing and set amount of samples for color rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(50)

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
        #sampled_target_objs = list(np.random.choice(target_objs, size=config['num_objects'], replace=False))
        sampled_target_objs = list(np.random.choice(target_objs, size=15, replace=False))

        light_point.set_energy(random.uniform(config["light_energy_max"], config["light_energy_min"]))

        # Randomize materials and set physics
        for obj in (sampled_target_objs):
            obj = set_material_properties(obj, cc_textures, randomize=True)

            bbox = obj.get_bound_box()
            min_coords = np.min(bbox, axis=0)
            max_coords = np.max(bbox, axis=0)
            diagonal = np.linalg.norm(max_coords - min_coords)

            target_diagonal = random.uniform(config["max_object_scaling"], config["min_object_scaling"])
            scaling_factor = (target_diagonal / diagonal)
            print("scaling factor: ", scaling_factor)

            obj.set_scale([scaling_factor,scaling_factor,scaling_factor])

        # Sample two light sources
        light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
        light_plane.replace_materials(light_plane_material)
        light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
        location = bproc.sampler.shell(center = [0, 0, 0], radius_min = config["cam"]["radius_min"], radius_max = config["cam"]["radius_max"],
                                elevation_min = config["cam"]["elevation_min"], elevation_max = config["cam"]["elevation_max"])
        light_point.set_location(location)

        # sample CC Texture and assign to room planes
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)
       

        bproc.object.sample_poses(objects_to_sample = sampled_target_objs,
                                    sample_pose_func = sample_pose_func, 
                                    max_tries = 100)
    
        # if i % 2 == 0:           
        #     bproc.object.sample_poses_on_surface(objects_to_sample=sampled_target_objs,
        #                                         surface=room_planes[0],
        #                                         sample_pose_func=sample_initial_pose,
        #                                         min_distance=0.1,
        #                                         max_distance=0.2)

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
                                    radius_min = config["cam"]["radius_min"],
                                    radius_max = config["cam"]["radius_max"],
                                    elevation_min = config["cam"]["elevation_min"],
                                    elevation_max = config["cam"]["elevation_max"])
            
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
                            dataset = "manibot_poop",
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
    config_path = "./manibot_cfg.yaml"

    dirname = os.path.dirname(__file__) #TODO

    with open(os.path.join(dirname, config_path), "r") as stream:
        config = yaml.safe_load(stream)

    render(config)


    
    

    