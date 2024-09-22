#!/usr/bin/env python3

import os
import glob
import shutil
import argparse
import json
from tqdm import tqdm

def transform_single_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    
    if len(data["system_measures"]) == 0:
        return
    elif "bottom" in data["system_measures"][0]:
        return
    
    # Process annotations
    new_annots = [[] for i in range(5)]
    for i, name in enumerate(["system_measures", "stave_measures", "staves", "systems", "grand_staff"]):
        for measure in data[name]:
            new_measure = {
                "left": measure["left"],
                "top": measure["top"],
                "right": measure["left"] + measure["width"],
                "bottom": measure["top"] + measure["height"]
            }
            new_annots[i].append(new_measure)

    # Construct the new JSON structure
    new_data = {
        "width": data["width"],
        "height": data["height"],
        "system_measures": new_annots[0],
        "stave_measures": new_annots[1],
        "staves": new_annots[2],
        "systems": new_annots[3],
        "grand_staff": new_annots[4]
    }

    # Save the processed JSON file
    with open(file_path, "w") as file:
        json.dump(new_data, file, indent=4)

def transform_directory(directory, verbose: bool = False):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                if verbose:
                    print(f"Processing {file_path}")
                transform_single_json_file(file_path)

def copy_files(src_dirs, dest_dir):
    # Ensure destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for src_dir in src_dirs:
        if not os.path.exists(src_dir):
            print(f"Source directory {src_dir} does not exist.")
            continue
        
        files = list(glob.glob(src_dir + "/**/*", recursive=True))

        for file in tqdm(files):
            if file.endswith(".json") or file.endswith(".png"):
                shutil.copy(file, dest_dir)

def clean_directory(directory):
    files = os.listdir(directory)
    
    # Create sets for json and png files
    json_files = {f[:-5] for f in files if f.endswith('.json')}
    png_files = {f[:-4] for f in files if f.endswith('.png')}
    
    # Find common filenames between json and png
    common_files = json_files & png_files
    
    # Delete files that are not common
    for f in files:
        name, ext = os.path.splitext(f)
        if (name not in common_files) or (ext not in ['.json', '.png']):
            os.remove(os.path.join(directory, f))
            print(f"Deleted: {f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy JSON and PNG files with matching names to a destination directory.")
    parser.add_argument("dest_dir", help="Destination directory to copy the files to.")
    parser.add_argument("src_dirs", nargs="+", help="Source directories containing JSON and PNG files.")
    args = parser.parse_args()

    # Copy
    copy_files(args.src_dirs, args.dest_dir)
    # Remove unmatched PNGs and JSONs
    clean_directory(args.dest_dir)
    # Transform JSONs to Pacha's format
    transform_directory(args.dest_dir)
