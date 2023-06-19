"""
e.g.
python smarter_data_selector.py --in_dir='../input/Gabriola_0630_2/06-30 Gabriola Panorama' --exclude_dir='../input/Gabriola_0630_2/exclude/' --out_dir='../output/TEST/' --n=200
"""
import argparse
from pathlib import Path
import pandas as pd
import math
import shutil
from PIL import Image
import numpy as np


def previous_annotation_filter(image_paths, annotated_image_dir):
    annotated_tile_paths = annotated_image_dir.glob('*')
    annotated_tile_ids = [t.name for t in annotated_tile_paths]
    boolean_filter = [tile_id(p) not in annotated_tile_ids for p in image_paths]
    return boolean_filter


def entropy_filter(image_paths, annotated_image_dir):
    # Find the mean & std of entropy for annotated images
    annotated_images = [Image.open(f) for f in annotated_image_dir.glob('*')]
    entropies = [i.entropy() for i in annotated_images]
    mean = np.mean(entropies)
    iqr = np.quantile(entropies, 0.75) - np.quantile(entropies, 0.25)
    min_val = mean - 1.5 * iqr
    max_val = mean + 1.5 * iqr

    # Filter images based on whether they are within 1.5 standard deviations of the mean
    boolean_filter = [min_val < Image.open(p).entropy() < max_val for p in image_paths]

    return boolean_filter


def tile_id(tile_path):
    return f"{tile_path.parent.name}.{tile_path.name}"


def filter_sample_pool(df, annotated_image_dir):
    df = df[previous_annotation_filter(df['image_paths'], annotated_image_dir)]
    df = df[entropy_filter(df['image_paths'], annotated_image_dir)]
    return df


def get_sample(in_dir, exclude_dir, sample_n, extension='jpg'):
    # Find all images
    all_images = list(in_dir.glob(f"*/*.{extension}")) + \
                 list(in_dir.glob(f"*.{extension}"))

    df = pd.DataFrame({'image_paths': all_images})
    df['parent_directory'] = df['image_paths'].apply(lambda x: x.parent)

    # Filter Images
    df = filter_sample_pool(df, exclude_dir)

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


def select_random_tiles(in_dir, exclude_dir, out_dir, n):
    sample_paths = get_sample(in_dir, exclude_dir, n)
    save_sample(sample_paths, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes a file path as input and tiles each .tif'
                                                 'file in that tif_directory.')
    parser.add_argument('--in_dir', help='File path to the tif_directory containing tifs to be tiled.')
    parser.add_argument('--out_dir', help='File path to output tif_directory.')
    parser.add_argument('--exclude_dir', help='File path to tif_directory containing images which have '
                                              'already been annotated')
    parser.add_argument('--n', type=int, help='Number of tiles to sample')
    args = parser.parse_args()

    select_random_tiles(in_dir=Path(args.in_dir),
                        exclude_dir=Path(args.exclude_dir),
                        out_dir=Path(args.out_dir),
                         n=args.n)

