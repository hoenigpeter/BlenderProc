import blenderproc as bproc
import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from blenderproc.python.utility.CollisionUtility import CollisionUtility
import random
import logging
import glob
from blenderproc.python.loader.ShapeNetLoader import _ShapeNetLoader
import glob
import json
import sys
import bmesh
import time
import bpy
import argparse
import shutil
from pathlib import Path


from pathlib import Path

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Customize the log format
    filename='app.log',  # Log to a file (optional)
    filemode='a'  # Overwrite the file each run (use 'a' for append)
)

def position_and_align_object(obj, robot):
    """
    Aligns an object to match the orientation of the robot's hand/gripper,
    with the longest dimension pointing upward and the smallest dimension
    between the fingers.
    
    Parameters:
    obj: blenderproc.object.Object
        The object to align
    robot: blenderproc.loader.URDFLoader
        The loaded robot model
    """
    # Get the hand's transformation matrix
    joint_positions = robot.get_all_local2world_mats()
    
    # Get finger positions for gripper orientation
    l_finger_pos = joint_positions[35][:3, 3]  # Left finger position
    r_finger_pos = joint_positions[41][:3, 3]  # Right finger position
    
    # Calculate grasp direction (between fingers)
    grasp_direction = l_finger_pos - r_finger_pos
    grasp_direction = grasp_direction / np.linalg.norm(grasp_direction)
    
    # Define world up vector
    world_up = np.array([0, 0, 1])
    
    # Get object's bounding box
    bbox = obj.get_bound_box()
    bbox_np = np.array(bbox)
    
    # Calculate bounding box dimensions
    min_corner = np.min(bbox_np, axis=0)
    max_corner = np.max(bbox_np, axis=0)
    dimensions = max_corner - min_corner
    
    # Find dimension indices sorted by size
    dim_indices = np.argsort(dimensions)
    min_dim_idx = dim_indices[0]  # Smallest (for grasping)
    mid_dim_idx = dim_indices[1]  # Middle
    max_dim_idx = dim_indices[2]  # Largest (should point up)
    
    # Step 1: First align the smallest dimension with grasp direction
    rotation_matrix = np.eye(3)
    rotation_matrix[:, min_dim_idx] = grasp_direction
    
    # Step 2: Project world up onto the plane perpendicular to grasp direction
    up_proj = world_up - np.dot(world_up, grasp_direction) * grasp_direction
    up_proj = up_proj / np.linalg.norm(up_proj)
    
    # This will be aligned with the longest dimension
    rotation_matrix[:, max_dim_idx] = up_proj
    
    # Step 3: Complete the right-handed coordinate system for the middle dimension
    middle_axis = np.cross(up_proj, grasp_direction)
    middle_axis = middle_axis / np.linalg.norm(middle_axis)
    rotation_matrix[:, mid_dim_idx] = middle_axis
    
    # Ensure the rotation matrix is orthogonal
    u, _, vh = np.linalg.svd(rotation_matrix)
    rotation_matrix = u @ vh
    
    # Position object between fingers
    center_grasp_pos = (l_finger_pos + r_finger_pos) / 2
    obj.set_location(center_grasp_pos)
    
    # Apply rotation to object
    obj.set_rotation_mat(rotation_matrix)
    
    return obj.get_local2world_mat(),dimensions[min_dim_idx]

def scale_object_to_hand(obj, robot):
    """
        Given the object and the robot, it scales the object accordingly to the graps size. 
        To do so, it calculates the distance between the fingers and scales the object to match that distance multiplied by
        a random factor between 0.05 and 0.5.
        Furthermore, if the object's height is too big, it scales it accordingly. In this case the object does not preserve its aspect ratio.
    """
    max_height = 0.15
    bbox_3d = obj.get_bound_box()
    bbox_3d = np.array(bbox_3d)
    min_corner = np.min(bbox_3d, axis=0)
    max_corner = np.max(bbox_3d, axis=0)
    dimensions = max_corner - min_corner  # Size along each local axis
    
    joint_positions = robot.get_all_local2world_mats()
            
    joint_obj = robot.get_all_visual_objs()[15]

    # Middle of fingers
    l_finger_pos = joint_positions[35][:3, 3]
    r_finger_pos = joint_positions[41][:3, 3]

    maximum_width = np.linalg.norm(l_finger_pos - r_finger_pos)

    scale = [maximum_width/min(dimensions)*random.uniform(0.05,.5) for _ in range(3)]
    print(scale)
    if max(dimensions)>max_height:
        scale_max = max_height/max(dimensions)*random.uniform(0.5,1)
        scale[np.argmax(dimensions)] = scale_max

    obj.set_scale(scale)

def randomize_viewing(robot):
    """
        Randomizes some of the robot's joints to get a different view of the object.
    """
    # arm_roll_joint 25
    # get random roation between -.8 and .8
    rotation = np.random.uniform(-0.8, 0.8)
    robot.set_rotation_euler_fk(link=robot.links[25], rotation_euler=rotation, mode='absolute')

    
    # head_pan_joint
    rotation = np.random.uniform(-.25 , 0)
    robot.set_rotation_euler_fk(link=robot.links[13], rotation_euler=rotation, mode='absolute')


    # head_tilt_joint
    rotation = np.random.uniform(-.3, 0)
    robot.set_rotation_euler_fk(link=robot.links[14], rotation_euler=rotation, mode='absolute')

    # arm_flex_joint
    rotation = np.random.uniform(-.5, 0)
    robot.set_rotation_euler_fk(link=robot.links[24], rotation_euler=rotation, mode='absolute')

def convert_to_convex_hull(obj):
    """
        Given an object, it calculates a convex hull and replaces the object by it.
    """
    obj.edit_mode()
    me = obj.get_mesh()
    bm = bmesh.new()
    bm.from_mesh(me)
    me = bpy.data.meshes.new("%s convexhull" % me.name)
    ch = bmesh.ops.convex_hull(bm, input=bm.verts)
    try:
        bmesh.ops.delete(
            bm,
            geom=ch["geom_unused"] + ch["geom_interior"],
            context='VERTS',
        )
    except:
        try:
            bmesh.ops.delete(
                bm,
                geom=ch["geom_unused"],
                context='VERTS',
            )
        except:
            try:
                bmesh.ops.delete(
                    bm,
                    geom=ch["geom_interior"],
                    context='VERTS',
                )
            except:
                logging.warning('Could not create convex hull')
                print('ERROR:Could not create convex hull')
                return False
            # except:
            #     raise ValueError('Could not create convex hull')
            #     except:
        # pass

    obj.object_mode()
    obj.update_from_bmesh(bm)
    return True


def get_objs(shapenet_path, cc_textures, type='shapenet', num_objects=1, categories_file = None):
    """
    Load objects from ShapeNet or BOP dataset, apply random textures, and set physics properties.
    Parameters:
    shapenet_path (str): Path to the ShapeNet dataset.
    cc_textures (list): List of textures to apply to the objects.
    type (str): Type of dataset to load ('shapenet' or other). Default is 'shapenet'.
    num_objects (int): Number of objects to load. Default is 1.
    categories_file (str): Path to the categories file containing ShapeNet IDs. Default is None.
    Returns:
    tuple: A tuple containing:
        - objs (list): List of loaded objects.
        - cc_textures (list): List of textures applied to the objects.
        - objs_bm (list): List of loaded objects for convex hull generation.
    """
    if num_objects == 0:
        return [], [], []
    objs = []
    objs_bm = []
    if type == 'shapenet':
        # open categories file, which has format ID NAME and get list of ID
        with open(categories_file, 'r') as f:
            categories = f.readlines()
            categories = [category.split()[0] for category in categories]

        # intersect categories with the ones in the filed']
        shapenet_categories = os.listdir(shapenet_path)
        categories = list(set(categories).intersection(shapenet_categories))

        # get relevant obj files
        obj_files = []
        taxonomy_file = shapenet_path+"/taxonomy.json"
        taxonomy_file = json.load(open(taxonomy_file))
        for category in categories:
            id_path = shapenet_path+"/"+category
            parent_id = _ShapeNetLoader.find_parent_synset_id(data_path=shapenet_path, synset_id=category,json_data=taxonomy_file)
            obj_files.extend(glob.glob(os.path.join(id_path, "*", "models", "model_normalized_scaled.ply")))
        # sample num_objects objects

        # We load each object two times; one of the two insances will be used for the convex hull
        if num_objects>len(obj_files):
            for file in obj_files:
                obj = bproc.loader.load_obj(file)[0]
                objs.append(obj)
                objs_bm.append(bproc.loader.load_obj(file)[0])
            objs = np.random.choice(objs, int(num_objects), replace=True)
            objs_bm = np.random.choice(objs_bm, int(num_objects), replace=True)
        else:
            obj_files= np.random.choice(obj_files, int(num_objects), replace=False)
            print(obj_files)
            for file in obj_files:
                obj = bproc.loader.load_obj(file)[0]
                objs.append(obj)
                objs_bm.append(bproc.loader.load_obj(file)[0])
            
        logging.info(f'Loaded {len(objs)} objects')
    else:
        bop_parent_path = "/ssd3/datasets_bop"

        objs = bproc.loader.load_bop_objs(bop_dataset_path = Path(bop_parent_path) / 'ycbv', mm2m = True)
    
        # sample args.num_objects objects
        objs = np.random.choice(objs, int(num_objects))

        for obj in (objs):
            obj.set_shading_mode('auto')
            obj.hide(True)
    
    # We get the convex hulls
    for idx,obj in enumerate(objs_bm):
        success = convert_to_convex_hull(obj)
        if not success:
            # remove obj frm objs_bm and also the obj from objs
            objs_bm.pop(idx)
            objs.pop(idx)
    
    # We hide them
    for obj in (objs):
        obj.set_shading_mode('auto')
        obj.hide(True)

    for obj in objs_bm:
        obj.hide(True)        



    # Randomize materials and set physics
    for obj in (objs):        
        random_cc_texture = np.random.choice(cc_textures)
        obj.replace_materials(random_cc_texture)
        if not obj.has_uv_mapping():
            obj.add_uv_mapping("smart")
        mat = obj.get_materials()[0]
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Alpha", 1.0)
        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.hide(True)
  

    return objs,cc_textures, objs_bm

def get_robot(urdf_file):
    """
    Loads a robot from a URDF file, modifies its materials, and returns the robot object.

    Args:
        urdf_file (str): The file path to the URDF file of the robot.

    Returns:
        bproc.loader.URDF: The loaded and modified robot object.

    The function performs the following steps:
    1. Loads the robot from the specified URDF file.
    2. Removes the link at index 0.
    3. Sets ascending category IDs for the robot's links.
    4. Iterates through the robot's links and modifies the materials of the visuals:
        - Sets the "Metallic" shader value to a random value between 0 and 0.1.
        - Sets the "Roughness" shader value to a random value between 0 and 0.5.
    5. Modifies the materials of specific links (indices 32, 34, 38, 40) to have a random black color.
    """
    print(urdf_file)
    try:
        robot = bproc.loader.load_urdf(urdf_file)
    except:
        for _ in range(5):
            # wait 10s and try again
            time.sleep(10)
            robot = bproc.loader.load_urdf(urdf_file)
    robot.remove_link_by_index(index=0)
    robot.set_ascending_category_ids()
    for link in robot.links:
        if link.visuals and hasattr(link.visuals[0], "materials"):
            materials = link.visuals[0].get_materials()
            for mat in materials:
                mat.set_principled_shader_value("Metallic", np.random.uniform(0, 0.1))
                # mat.set_principled_shader_value("Specular", np.random.uniform(0, 0.5))
                mat.set_principled_shader_value("Roughness", np.random.uniform(0, 0.5))

    # Specify the indices of the links you want to modify
    indices_to_modify = [32, 34, 38, 40]  # Replace with the desired indices

    for index in indices_to_modify:
        link = robot.links[index]
        materials = link.visuals[0].get_materials()
        for mat in materials:
            black_col = np.random.uniform(0.001, 0.02) 
            mat.set_principled_shader_value("Base Color", [black_col, black_col, black_col, 1]) 
            # mat.set_principled_shader_value("Specular", np.random.uniform(0, 0.3))
    return robot


def prepare_room():
    """
    Prepares a room environment with lighting for a simulation.

    This function creates a ceiling light plane, a point light, and the walls of a room using 
    the bproc library. The ceiling light plane is assigned a material, and the point light's 
    energy is set. The room is constructed using planes positioned and rotated to form walls.

    Returns:
        tuple: A tuple containing the following elements:
            - light_plane (bproc.object): The ceiling light plane object.
            - light_plane_material (bproc.material): The material assigned to the ceiling light plane.
            - light_point (bproc.types.Light): The point light object.
            - room_planes (list of bproc.object): A list of plane objects representing the walls of the room.
    """
    # sample light color and strength from ceiling
    light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')

    # sample point light on shell
    light_point = bproc.types.Light()
    light_point.set_energy(50)

    # create room
    room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
                    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
                    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
                    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
                    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)

    return light_plane, light_plane_material, light_point, room_planes
def is_point_inside_bbox(point, bbox):
    """
    Check if a point is inside a bounding box.

    Parameters:
    -----------
    point : array-like
        The point to check
    bbox : array-like
        The bounding box to check againstshapen

    Returns:
    --------
    bool
        True if the point is inside the bounding box, False otherwise
    """
    bbox = np.array(bbox)
    min_corner = np.min(bbox, axis=0)
    max_corner = np.max(bbox, axis=0)
    return np.all(min_corner <= point) and np.all(point <= max_corner)


def generate_scene(
    urdf_file="examples/resources/medical_robot/miro.urdf", 
    output_dir="examples/advanced/urdf_loading_and_manipulation/output", 
    num_samples=1,
    include_object = True, 
    shapenet_path="ssd3/datasets_bop/shapenetcorev2/models_bop-renderer_scale=0.1",
    cc_textures_path="/home/pmarg/BlenderProc/resources/cc_textures",
    dataset_type='bop',
    resolution_X = 640,
    resolution_Y = 480,
    random_categories=None,
    render=False
):
    """
    Generate a scene with a robot, objects, and rendering.

    Parameters:
    -----------
    urdf_file : str, optional
        Path to the URDF file for the robot
    output_dir : str, optional
        Directory to save output files
    num_objects : int, optional
        Number of objects to sample
    shapenet_path : str, optional
        Path to ShapeNet dataset
    cc_textures_path : str, optional
        Path to CC Textures
    dataset_type : str, optional
        Type of dataset to load objects from ('shapenet' or 'bop')
    random_categories : list, optional
        List of categories to sample objects from (for ShapeNet)

    Returns:
    --------
    dict
        Rendered scene data
    """
    # Initialize BlenderProc

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)


    logging.info('Started Rendering script')
    # Load CC Textures
    cc_textures = bproc.loader.load_ccmaterials(cc_textures_path)
    print("CC Textures loaded")

    # Load robot
    robot = get_robot(urdf_file)
    print("Robot loaded")
    # Prepare room
    light_plane, light_plane_material, light_point, room_planes = prepare_room()

    # Set rendering parameters
    bproc.camera.set_resolution(int(640), int(480))
    bproc.renderer.enable_depth_output(True)

    # Set up lighting
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3,6), 
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0])
    )  
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(
        center=[0, 0, 0], 
        radius_min=2, 
        radius_max=3,
        elevation_min=5, 
        elevation_max=89
    )
    light_point.set_location(location)
    
    if include_object:
        # Load objects
        objs, cc_textures,objs_ch = get_objs(shapenet_path, cc_textures, type=dataset_type, num_objects=num_samples,categories_file='examples/datasets/bop_challenge/categories.txt')
        print("Objects loaded")

        # Process each object
        for obj, obj_ch in zip(objs, objs_ch):
            random_cc_texture = np.random.choice(cc_textures)
            for plane in room_planes:
                plane.replace_materials(random_cc_texture)
            
            print("Starting object ", obj)

            # Flex wrist for better view
            robot.set_rotation_euler_fk(link=robot.links[26], rotation_euler=-1.0, mode='absolute')
            robot.set_rotation_euler_fk(link=robot.links[32], rotation_euler=0.8, mode='absolute')
            robot.set_rotation_euler_fk(link=robot.links[38], rotation_euler=0.8, mode='absolute')
            
            if dataset_type == 'shapenet':
                scale_object_to_hand(obj, robot)
            
            randomize_viewing(robot)

            joint_positions = robot.get_all_local2world_mats()

            # Get Grasp position
            l_finger_pos = joint_positions[35][:3, 3]
            r_finger_pos = joint_positions[41][:3, 3]
            center_grasp_pos = (l_finger_pos + r_finger_pos) / 2
            distance = np.linalg.norm(l_finger_pos - r_finger_pos)
            obj.hide(False)
            obj.set_location(center_grasp_pos)
            rot_step = -0.005/2

            local2world, obj_width = position_and_align_object(obj, robot)

            obj_ch.hide(False)
            obj_ch.set_scale(obj.get_scale())
            obj_ch.set_location(local2world[:3,3])
            obj_ch.set_rotation_mat(local2world[:3,:3])
            has_collided = False
            if obj_width>1.5*np.linalg.norm(l_finger_pos - r_finger_pos):
                print("Object too large")
                logging.warning('COULD NOT PLACE OBJECT; TOO LARGE')
            else:    
                # We check if the object is inside the robot's hand camera and move it away from it
                joint_positions = robot.get_all_local2world_mats()
                joint_obj = robot.get_all_visual_objs()[21]

                joint_middle_vec = center_grasp_pos-joint_obj.get_location()
                step_move = 0.05
                iter = 1
                link_bvh_tree = joint_obj.create_bvh_tree()

                while CollisionUtility.check_mesh_intersection(joint_obj,obj) and CollisionUtility.is_point_inside_object(point=joint_obj.get_location(),obj_bvh_tree=link_bvh_tree,obj=obj) and  iter<1/iter*1.5:
                    obj.set_location(obj.get_location() + joint_middle_vec*step_move)
                    iter+=1

                print(f'Moved object {iter*step_move} away from camera')
                logging.info(f'Moved object {iter*step_move} ')
                bvh_tree_ch = obj_ch.create_bvh_tree()



                has_collided={32:False, 38:False}
                rotations = 0
                bvh_tree = obj.create_bvh_tree()

                # We progressively move each finger to check for collisions with the object            
                while rotations < 1000 and not (has_collided[32] and has_collided[38]):
                    joint_positions = robot.get_all_local2world_mats()
                    # middle of fingers
                    l_finger_pos = joint_positions[35][:3, 3]
                    r_finger_pos = joint_positions[41][:3, 3]
                    distance = np.linalg.norm(l_finger_pos - r_finger_pos)

                    finger_positions = {32:joint_positions[35], 38:joint_positions[41]}

                    for finger_idx,finger_pos in finger_positions.items():
                        if has_collided[finger_idx]:
                            continue
                        point = finger_pos[:3, 3]
                        collision = CollisionUtility.is_point_inside_object(obj=obj_ch, obj_bvh_tree=bvh_tree_ch, point=point)
                        if collision:
                            print(f'Collision detected with finger {finger_idx} at iteration {rotations}')
                            has_collided[finger_idx] = True
                        with open(os.devnull, 'w') as fnull:
                            sys.stdout = fnull
                            robot.set_rotation_euler_fk(link=robot.links[finger_idx], rotation_euler=-0.005, mode='relative')
                            sys.stdout = sys.__stdout__
                    rotations += 1
                print(f'Rotated {rotations} times, collision: {has_collided}')
                logging.info(f'Rotated {rotations} times')

                obj_ch.hide(True)

                # We set up the camera                
                joint_positions = robot.get_all_local2world_mats()

                # Sample camera pose
                location = joint_positions[21][:3, 3]

                poi = bproc.object.compute_poi(robot.links[42].get_visuals())
                rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
                cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
                bproc.camera.add_camera_pose(cam2world_matrix)
                if render:
                    print("\n\n\nRENDERING\n-----------------------------------------")
                    # Render data
                    data = bproc.renderer.render()

                    # Write BOP format data
                    bproc.writer.write_bop(
                        Path(output_dir) / 'bop_data' / 'positive',
                        target_objects=[robot.links[1]],
                        depths=data["depth"],
                        colors=data["colors"], 
                        m2mm=False
                    )
                    logging.info("Object rendered successfully")
                else:
                    return
                # print 10 empty liens and -- to separate objects
                print("--"*10)
                print("\n"*10)

            obj.hide(True)

            bproc.utility.reset_keyframes()
    else:
        for _ in range(num_samples):
            random_cc_texture = np.random.choice(cc_textures)
            for plane in room_planes:
                plane.replace_materials(random_cc_texture)
            # Flex wrist for better view
            robot.set_rotation_euler_fk(link=robot.links[26], rotation_euler=-1.0, mode='absolute')
            robot.set_rotation_euler_fk(link=robot.links[32], rotation_euler=0.0, mode='absolute')
            robot.set_rotation_euler_fk(link=robot.links[38], rotation_euler=0.0, mode='absolute')
            
            randomize_viewing(robot)
            joint_positions = robot.get_all_local2world_mats()
            location =joint_positions[21][:3, 3]
            poi = bproc.object.compute_poi(robot.links[42].get_visuals())
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            bproc.camera.add_camera_pose(cam2world_matrix)
            if render:
                print("\n\n\nRENDERING\n-----------------------------------------")
                # Render data
                data = bproc.renderer.render()

                # Write BOP format data
                bproc.writer.write_bop(
                    Path(output_dir) / 'bop_data' / 'negative',
                    target_objects=[robot.links[1]],
                    depths=data["depth"],
                    colors=data["colors"], 
                    m2mm=False
                )
                logging.info("Object rendered successfully")
            else:
                return
            # print 10 empty liens and -- to separate objects
            print("--"*10)
            print("\n"*10)
            bproc.utility.reset_keyframes()



# Optionally, keep the main block for direct script execution
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate data for the BOP challenge')
    parser.add_argument('--render', action='store_true', help='Render the scene')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--positive_ratio', type=float, default=0.5, help='Ratio of positive samples')
    args = parser.parse_args()

    # add the path to load the file
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))    
    # load config.json
    with open('examples/datasets/bop_challenge/config.json') as f:
        config = json.load(f)
    print("---"*10)
    print("STARTING")
    print("---"*10)
    print("Args:"+str(args))
    print("Config:")
    print(config)
    print("---"*10)
    logging.info(f'Started Rendering script:\n - number of samples {args.num_samples} and ratio {args.positive_ratio}\nOther args: '+str(config))
    positive_samples = int(args.positive_ratio*args.num_samples)
    negative_samples = args.num_samples-positive_samples
    # positive
    bproc.init()
    if positive_samples>0:
        generate_scene(
            urdf_file=config['urdf_file'], 
            output_dir=config['output_dir'],    
            num_samples=positive_samples,
            include_object = True,
            shapenet_path=config['shapenet_path'],
            cc_textures_path=config['textures_path'],
            dataset_type=config['dataset_type'],
            render = args.render
        )

    if negative_samples>0:
    # negative
        generate_scene(
            urdf_file=config['urdf_file'], 
            output_dir=config['output_dir'],    
            num_samples=negative_samples,
            include_object = False,
            shapenet_path=config['shapenet_path'],
            cc_textures_path=config['textures_path'],
            dataset_type=config['dataset_type'],
            render = args.render

        )
    # move files in output_dir/bop_data/positive/train_pbr/000000/ to dataset_output_path/positive/ 
    # create the folder if it does not exist
    os.makedirs(config['dataset_output_path']+'/positive', exist_ok=True)
    os.makedirs(config['dataset_output_path']+'/negative', exist_ok=True)
    # copy all positive files at once using shutil
   
    positive_src = Path(config['output_dir']) / 'bop_data' / 'positive' / 'train_pbr' / '000000' / 'rgb'
    # check that the folder exists
    if positive_samples>0 and positive_src.exists():
        
        positive_dst = Path(config['dataset_output_path']) / 'positive'
        positive_dst.mkdir(parents=True, exist_ok=True)
        for src_file in positive_src.iterdir():
            dst_file = positive_dst / src_file.name
            if not dst_file.exists():
                shutil.copy(src_file, dst_file)
    # copy all negative files at once using shutil
    negative_src = Path(config['output_dir']) / 'bop_data' / 'negative' / 'train_pbr' / '000000' / 'rgb'
    if negative_samples>0 and negative_src.exists():
        negative_dst = Path(config['dataset_output_path']) / 'negative'
        negative_dst.mkdir(parents=True, exist_ok=True)
        for src_file in negative_src.iterdir():
            dst_file = negative_dst / src_file.name
            if not dst_file.exists():
                shutil.copy(src_file, dst_file)