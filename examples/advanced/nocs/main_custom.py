import blenderproc as bproc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('shapenet_path', help="Path to the downloaded shape net core v2 dataset, get it from http://www.shapenet.org/")
parser.add_argument('output_dir', nargs='?', default="examples/advanced/nocs/output", help="Path to where the final files, will be saved")
args = parser.parse_args()

bproc.init()


# load the ShapeNet object into the scene
shapenet_obj = bproc.loader.load_shapenet(args.shapenet_path, used_synset_id="02942699", used_source_id="97690c4db20227d248e23e2c398d8046", move_object_origin=False)
shapenet_obj = bproc.loader.load_shapenet(args.shapenet_path, used_synset_id="02942699", used_source_id="97690c4db20227d248e23e2c398d8046", move_object_origin=False)

#/hdd/datasets_bop/shapenetcorev2/models_orig/02942699/97690c4db20227d248e23e2c398d8046/models/model_normalized.obj
#/hdd2/real_camera_dataset/obj_models/train

# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([5, -5, 5])
light.set_energy(1000)

bproc.camera.set_resolution(256, 256)

number_of_scenes = 1

for scene_id in range(number_of_scenes):
    # Sample five camera poses
    for i in range(25):
        # Sample random camera location around the object
        location = bproc.sampler.sphere([0, 0, 0], radius=2, mode="SURFACE")
        # Compute rotation based on vector going from location towards the location of the ShapeNet object
        rotation_matrix = bproc.camera.rotation_from_forward_vec(shapenet_obj.get_location() - location)
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)

    # Render RGB images
    data = bproc.renderer.render()
    bproc.writer.write_images(args.output_dir + "/rgb", scene_id, data)

    # Render NOCS
    data.update(bproc.renderer.render_nocs())
    bproc.writer.write_images(args.output_dir + "/nocs", scene_id, data)
