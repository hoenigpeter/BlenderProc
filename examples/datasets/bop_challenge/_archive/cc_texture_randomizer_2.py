import blenderproc as bproc
import argparse
import os
import numpy as np
import shutil
import random

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Copy a specified number of textures from source to destination.')
parser.add_argument('source_dir', type=str, help='The source directory containing the textures')
parser.add_argument('dest_dir', type=str, help='The destination directory to copy the textures to')
parser.add_argument('--num_textures', type=int, default=5, help='The number of textures to copy in each repeat')
parser.add_argument('--num_repeats', type=int, default=30, help='The number of times to repeat the copying process')

args = parser.parse_args()

# Load textures
cc_textures = bproc.loader.load_ccmaterials(args.source_dir)

# Get a list of all the folder names in the source directory
folder_names = [texture.blender_obj.name for texture in cc_textures]

# Initialize a counter for the new folder names
folder_counter = 0

# Loop through the number of repeats
for _ in range(args.num_repeats):
    # Randomly select num_textures folder names from the list
    selected_folder_names = random.sample(folder_names, args.num_textures)

    # Create a new folder in the destination directory with a unique name
    dest_folder_name = str(folder_counter)
    dest_folder_path = os.path.join(args.dest_dir, dest_folder_name)
    os.makedirs(dest_folder_path, exist_ok=True)

    # Loop through the selected folder names and copy them from the source to the destination directory
    for folder_name in selected_folder_names:
        source_path = os.path.join(args.source_dir, folder_name)
        dest_path = os.path.join(dest_folder_path, folder_name)
        shutil.copytree(source_path, dest_path)

        # Remove the folder from the list of folder names to prevent it from being selected again
        #folder_names.remove(folder_name)

    folder_counter += 1

# Example command to run the script
# python script.py /home/hoenig/BlenderProc/resources/cc_textures /home/hoenig/BlenderProc/examples/datasets/bop_challenge/cc_textures_5r --num_textures=5 --num_repeats=30
