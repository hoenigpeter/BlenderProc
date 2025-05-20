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

from scipy.spatial.transform import Rotation
import webdataset as wds
import tarfile

def add_nocs_images_to_tar(tar_path, nocs_dir):
    """
    Adds all .nocs.png files from nocs_dir into the tar archive at tar_path.
    """
    print("tar_path: ", tar_path)
    print("nocs_dir: ", nocs_dir)
    print()
    with tarfile.open(tar_path, "a") as tar:
        print(f"Adding files from {nocs_dir} to {tar_path}")
        for file_name in os.listdir(nocs_dir):
            print(file_name)
            if file_name.endswith(".nocs.png"):
                full_path = os.path.join(nocs_dir, file_name)
                tar_name = file_name  # or f"{file_name}" if you want a subfolder inside the tar
                tar.add(full_path, arcname=tar_name)
                print(f"Added {full_path} to {tar_path} as {tar_name}")

if __name__ == "__main__":
    
    bproc.init()
    for num in range(1):
        output_dir = "./megapose_nocs_gso"

        dirname = os.path.dirname(__file__)
        #render(output_dir, num)
        add_nocs_images_to_tar(f"/hdd/megapose_gso/{num:08d}.tar", output_dir + "/" + f"{num:08d}")