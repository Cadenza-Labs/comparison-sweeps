import os
import shutil
import sys

def selective_copy(src, dst):
    """
    Copy the folder structure from src to dst, but only keep files named
    eval.csv or layer_ensembling.csv.

    Parameters:
    - src: Source directory path
    - dst: Destination directory path
    """
    for dirpath, dirnames, filenames in os.walk(src):
        for filename in filenames:
            if filename in ["cfg.yaml", "eval.csv", "layer_ensembling.csv"]:
                src_file = os.path.join(dirpath, filename)
                # Compute the relative path to maintain folder structure
                relative_path = os.path.relpath(dirpath, src)
                dst_dir = os.path.join(dst, relative_path)
                os.makedirs(dst_dir, exist_ok=True) # Ensure the directory exists
                dst_file = os.path.join(dst_dir, filename)
                shutil.copy2(src_file, dst_file) # Copy the file to destination

if __name__ == "__main__":
    src_directory = sys.argv[1]
    dst_directory = sys.argv[2]
    selective_copy(src_directory, dst_directory)
