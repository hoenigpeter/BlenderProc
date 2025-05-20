import argparse
import os
import numpy as np
import yaml
import sys
import imageio
import random
import shutil
import glob
import json

def reconstruct_category_mapping(mesh_dir):
    subfolders = sorted([d for d in os.listdir(mesh_dir) if os.path.isdir(os.path.join(mesh_dir, d))])
    
    category_id_map = {}
    i = 0  # Counter used in load_meshes

    for subfolder in subfolders:
        try:
            obj_id = int(subfolder)  # Convert folder name to integer directly
        except ValueError:
            print(f"Warning: Unexpected folder '{subfolder}', skipping.")
            continue

        subfolder_path = os.path.join(mesh_dir, subfolder)
        category_folders = sorted(os.listdir(subfolder_path))
        
        for category_folder in category_folders:
            category_id_map[i] = obj_id  # Map category index i to the actual object ID
            i += 1

    with open("category_mapping.json", "w") as f:
        json.dump(category_id_map, f, indent=4)

    return category_id_map

objects = [
    {"id": 0, "name": "bottle", "min_diagonal": 0.2, "max_diagonal": 0.25},
    {"id": 1, "name": "bowl", "min_diagonal": 0.15, "max_diagonal": 0.25},
    {"id": 2, "name": "camera", "min_diagonal": 0.15, "max_diagonal": 0.25},
    {"id": 3, "name": "can", "min_diagonal": 0.1, "max_diagonal": 0.2},
    {"id": 4, "name": "laptop", "min_diagonal": 0.3, "max_diagonal": 0.5},  # Opened laptop
    {"id": 5, "name": "mug", "min_diagonal": 0.1, "max_diagonal": 0.18}
]

target_objs = reconstruct_category_mapping("/ssd3/real_camera_dataset/obj_models/train")
