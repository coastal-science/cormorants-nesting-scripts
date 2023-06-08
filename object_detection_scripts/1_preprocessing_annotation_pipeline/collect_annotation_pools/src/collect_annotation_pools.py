"""
Setting up inputs
* Original Image Directory. e.g. from ../input/snb/cormorants-nesting-scripts/SNB_20_24062021 run
  ln -s ../../../../tile_tifs/output/SNB_20_24062021/* .
* Annotated Image Directories. e.g. from ../input/snb1/anno_snb1 run
  ln -s ../../../../random_data_selector/output/SNB/* .

e.g.
python3 collect_annotation_pools.py \
  --original_image_dir ../input/gab2/06-30\ Gabriola\ Panorama/ \
  --annotated_image_dirs ../input/gab2/ANNO_ROUND1/ ../input/gab2/ANNO_ROUND2/ \
  --out_dir ../output/gab2
"""
import argparse
from pathlib import Path


def tile_id(tile_path):
    return f"{tile_path.parent.name}.{tile_path.name}"


def get_original_image_list(original_pool_dir):
    all_images = original_pool_dir.glob("*")
    return list(all_images)


def get_annotated_image_list(annotated_pool_dirs):
    all_annotated_images = []

    for annotated_pool_dir in annotated_pool_dirs:
        all_annotated_images += list(annotated_pool_dir.glob("*"))

    return all_annotated_images


def find_unannotated_image_list(original_pool, annotated_pool):
    # Annotated IDs
    annotated_IDs = [a.name for a in annotated_pool]

    # For each image in the original pool, check if it is in the annotated_pool.
    unannotated = [o for o in original_pool if tile_id(o) not in annotated_IDs]

    return unannotated


def collect_annotation_pools(original_pool_dir, annotated_dirs, output_dir):
    # Find the original images
    original_pool = get_original_image_list(original_pool_dir)

    # Find the images that have been annotated -- perhaps the easiest way to do this is to look at the tf records?
    annotated_pool = get_annotated_image_list(annotated_dirs)

    # Determine which images are part of the unannotated pool.
    unannotated_pool = find_unannotated_image_list(original_pool, annotated_pool)

    # Output two updated pools -- one should be the annotated pool (a directory of all annotated
    # images) and the other should be the unannotated pool (a directory of all the unannotated
    # images).
    annotated_pool_out_dir = output_dir.joinpath('annotated_pool')
    if not annotated_pool_out_dir.exists():
        annotated_pool_out_dir.mkdir()
        for annotated_image in annotated_pool:
            print(annotated_image.absolute())
            annotated_pool_out_dir.joinpath(annotated_image.name).symlink_to(annotated_image.absolute())
        print(f"Wrote {len(annotated_pool)} images to the pool of annotated images.")
    else:
        print(f"Wrote no images to pool of annotated images. Check that your output folder wasn't"
              f"already being used.")

    unannotated_pool_out_dir = output_dir.joinpath('unannotated_pool')
    if not unannotated_pool_out_dir.exists():
        unannotated_pool_out_dir.mkdir()
        for unannotated_image in unannotated_pool:
            unannotated_pool_out_dir.joinpath(unannotated_image.name).symlink_to(unannotated_image.absolute())
        print(f"Wrote {len(unannotated_pool)} images to the pool of unannotated images")
    else:
        print(f"Wrote no images to pool of unannotated images. Check that your output folder wasn't"
              f"already being used.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes a file path as input and tiles each .tif'
                                                 'file in that directory.')
    parser.add_argument('--original_image_dir', help='File path to the directory containing all'
                                                     'images in the dataset. (e.g. output of the'
                                                     'tile_tifs).')
    parser.add_argument('--annotated_image_dirs', nargs='+', help='.')
    parser.add_argument('--out_dir', help='.')

    args = parser.parse_args()

    collect_annotation_pools(original_pool_dir=Path(args.original_image_dir),
                             annotated_dirs=[Path(d) for d in args.annotated_image_dirs],
                             output_dir=Path(args.out_dir))