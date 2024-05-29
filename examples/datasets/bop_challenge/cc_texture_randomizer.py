import blenderproc as bproc
import argparse
import os
import numpy as np
import shutil
import random

# set the source and destination directories
source_dir = '/home/hoenig/BlenderProc/resources/cc_textures'
dest_dir = '/home/hoenig/BlenderProc/examples/datasets/bop_challenge/cc_textures_5r'

# set the number of folders to copy
num_folders_to_copy = 5

# set the number of times to repeat the process
num_repeats = 30

cc_textures = bproc.loader.load_ccmaterials("resources/cc_textures")

# get a list of all the folder names in the source directory
folder_names = []
for texture in cc_textures:
    folder_names.append(texture.blender_obj.name)
    print(texture.blender_obj.name)

# initialize a counter for the new folder names
folder_counter = 0

# loop through the number of repeats
for i in range(num_repeats):
    # randomly select num_folders_to_copy folder names from the list
    selected_folder_names = random.sample(folder_names, num_folders_to_copy)

    # create a new folder in the destination directory with a unique name
    dest_folder_name = str(folder_counter)
    dest_folder_path = os.path.join(dest_dir, dest_folder_name)
    os.makedirs(dest_folder_path)

    # loop through the selected folder names and copy them from the source to the destination directory
    for folder_name in selected_folder_names:
        source_path = os.path.join(source_dir, folder_name)

        # copy the folder from the source to the new folder in the destination directory
        dest_path = os.path.join(dest_folder_path, folder_name)
        print(dest_path)
        shutil.copytree(source_path, dest_path)

        # remove the folder from the list of folder names to prevent it from being selected again
        folder_names.remove(folder_name)

    folder_counter += 1
