import blenderproc as bproc
import numpy as np
import os
import bpy

scene = "/home/hoenig/BlenderProc/custom_scripts/test.blend"
output = "/home/hoenig/BlenderProc/custom_scripts/output"

bproc.init()

# load the objects into the scene
objs = bproc.loader.load_obj("/home/hoenig/BlenderProc/custom_scripts/models/obj_000001.ply")

for obj in objs:
    obj.set_cp("category_id", 1)
    obj.set_scale([0.001,0.001,0.001])

    mat = obj.get_materials()[0]
    mat.set_principled_shader_value("Roughness", 0)
    mat.set_principled_shader_value("Specular IOR Level", 0.0)
    #mat.set_principled_shader_value("Metallic", np.random.uniform(0.3, 1.0))
    mat.set_principled_shader_value("Alpha", 0.0)
    # grey_col = np.random.uniform(0.1, 0.7)   
    mat.set_principled_shader_value("Base Color", [0, 0, 0, 0])   
    # Update the object material in bpy directly

light = bproc.types.Light()
light.set_type("POINT")
light.set_location([5, -5, 5])
light.set_energy(1000)

# define the camera intrinsics
bproc.camera.set_resolution(640, 480)

# Set their location and rotation
for obj in objs:
    obj.set_location([0,0,0])
    obj.set_rotation_euler([1, 0, 0])

poi = bproc.object.compute_poi(objs)
location = np.array([0,0,0.2])

rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
# Add homog cam pose based on location an rotation
cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
bproc.camera.add_camera_pose(cam2world_matrix)

bpy.context.view_layer.update()

########## The Line causing all the problems ###############
bpy.data.scenes['Scene'].render.use_freestyle = True
bpy.ops.scene.freestyle_linestyle_new()

bpy.data.scenes['Scene'].render.line_thickness = 1

#freestyle_settings = bpy.data.scenes['Scene'].view_layers['ViewLayer'].freestyle_settings
#linesets = freestyle_settings.linesets['LineSet']
# linesets.select_silhouette = True
# linesets.select_crease = True
# linesets.select_border = True
#linesets.select_contour = True

bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.image_settings.file_format='PNG'
bpy.context.scene.render.image_settings.color_mode = 'RGBA'

bpy.context.scene.render.filepath='/home/hoenig/BlenderProc/custom_scripts/blender.png'
bpy.ops.render.render(write_still = True)