"""
e.g. python random_data_selector.py --in_dir='../input/' --out_dir='../output/' --n=200
"""
import argparse
from pathlib import Path
import pandas as pd
import math
import shutil


def get_sample(in_dir, sample_n, extension='jpg'):
    input_path = Path(in_dir)

    # Find all images
    all_images = list(input_path.glob(f"*/*.{extension}")) + \
                 list(input_path.glob(f"*.{extension}"))

    df = pd.DataFrame({'image_paths': all_images})
    df['parent_directory'] = df['image_paths'].apply(lambda x: x.parent)

    # Find number of tiles per folder
    n = math.ceil(sample_n / df['parent_directory'].nunique())

    # Sample images from each parent tif_directory
    grouped_sample = df.groupby('parent_directory').sample(n)

    # Total sample
    total_sample = grouped_sample['image_paths'].sample(sample_n)

    return total_sample


def save_sample(sample_paths, out_dir):
    out_path = Path(out_dir)
    if not out_path.exists():
        out_path.mkdir()
    for tile_path in sample_paths:
        tile_id = f"{tile_path.parent.name}.{tile_path.name}"
        new_tile_path = out_path.joinpath(tile_id)
        shutil.copy(tile_path, new_tile_path)


def select_random_tiles(in_dir, out_dir, n):
    sample_paths = get_sample(in_dir, n)
    save_sample(sample_paths, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes a file path as input and tiles each .tif'
                                                 'file in that tif_directory.')
    parser.add_argument('--in_dir', help='File path to the tif_directory containing tifs to be tiled.')
    parser.add_argument('--out_dir', help='File path to output tif_directory.')
    parser.add_argument('--n', type=int, help='Number of tiles to sample')
    args = parser.parse_args()

    select_random_tiles(Path(args.in_dir),
                        Path(args.out_dir),
                        args.n)

