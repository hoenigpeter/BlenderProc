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
from tqdm import tqdm
import re

import webdataset as wds
import tarfile

def main():

    for num in range(1050):
        print("Processing folder number:", num)
        output_directory = "/ssd3/megapose_gso_stripped"
        megapose_path = "/hdd/megapose_gso" 

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

        output_tar_path = os.path.join(output_directory, f"{num:08d}.tar")

        with wds.TarWriter(output_tar_path) as writer:            
            for instance_id, camera_data, object_data, infos in tqdm(dataset, desc="Rendering"):

                output_data = {
                    "__key__": instance_id,
                    "camera_data.json": camera_data,
                    "object_datas.json": object_data,
                    "infos.json": infos,
                }
                writer.write(output_data)

if __name__ == "__main__":
    main()