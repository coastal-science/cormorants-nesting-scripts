"""
Based off
github.com/jillianderson8/cmpt733finalproject/blob/master/Code/3_tileAnnotate/src/tileAnnotate.py

e.g. Tile a directory of tifs
python3 tile_tif.py \
--in_dir ../input/2020_SNB/ \
--out_dir ../output/2020_SNB/ \
--tile_height 1000 \
--tile_width 1000

e.g. Tile a single file
python3 tile_tif.py \
--in_file ../input/2020_SNB/FILENAME.tif \
--out_dir ../output/2020_SNB/FILENAME/ \
--tile_height 1000 \
--tile_width 1000
"""
import math
from PIL import Image, ImageFile
import argparse
from pathlib import Path

import pandas as pd
import geopandas as gpd
import shapely
import numpy as np
from PIL import ImageDraw

from shapely.geometry import Polygon, box
import ast

Image.MAX_IMAGE_PIXELS = 3000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


def tile_image(image, tile_height, tile_width, out_dir, out_extension, exclude_all_black_tiles=False):
    """
        Divide the given image into tiles
    """
    img_width, img_height = image.size

    # Create tile output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tile Images
    num_tiles_vert = math.ceil(img_height / tile_height)
    num_tiles_horiz = math.ceil(img_width / tile_width)
    for i in range(0, num_tiles_vert):
        result_height = min(img_height - i * tile_height, tile_height)
        for j in range(0, num_tiles_horiz):
            result_width = min(img_width - j * tile_width, tile_width)
            box = (j * tile_width, i * tile_height, j * tile_width + result_width,
                   i * tile_height + result_height)
            tile = image.crop(box)
            write_tile = True
            if exclude_all_black_tiles:
                if tile.getbbox() is None:
                    write_tile = False
            if write_tile:
                try:
                    # Save Tile Image
                    file_name = f"{i}.{j}.{out_extension}"
                    out_path = out_dir.joinpath(file_name)
                    tile.convert('RGB').save(out_path)

                except Exception as e:
                    print(e)


def tile_single_tif(file_path, tile_height, tile_width, out_dir, out_extension='JPG'):
    # Open Image
    image = Image.open(file_path)

    # Tile Image
    tile_image(image,
               tile_height=tile_height,
               tile_width=tile_width,
               out_dir=out_dir,
               out_extension='jpg')


def tile_all_tifs(directory, tile_height, tile_width, out_directory, out_extension='JPG'):
    all_tifs = directory.glob("*.tif")
    for tif in all_tifs:
        # Open Image
        image = Image.open(tif)

        # Tile Image
        tif_tile_dir = out_directory.joinpath(tif.stem)
        tile_image(image,
                   tile_height=tile_height,
                   tile_width=tile_width,
                   out_dir=tif_tile_dir,
                   out_extension='jpg')


# ============= MASKING =============
def find_full_canvas_dims(df):
    tile_ys, tile_xs = zip(*df['image'].transform(lambda x: Path(x).name.split('.')[:2]))
    tile_xs = [int(x) for x in tile_xs]
    tile_ys = [int(y) for y in tile_ys]
    return max(tile_xs) + 1, max(tile_ys) + 1


def load_mask(f, resize_dims=(74.576, 33.620)):
    df = pd.read_csv(f)
    print(f"Loading mask for {df['img.img_path'].iloc[0]}")
    mask_points = [(d['x'], d['y']) for d in ast.literal_eval(df['anno.data'].iloc[0])]
    resized_points = [(x*resize_dims[0], y*resize_dims[1]) for x, y in mask_points]
    mask_geom = shapely.geometry.Polygon(resized_points)
    return mask_geom


def create_detection_geom(tile_name, detection_box):
    y_tile, x_tile = Path(tile_name).name.split('.')[:2]
    y_tile = int(y_tile)
    x_tile = int(x_tile)

    y1, x1, y2, x2 = detection_box  # format: [y1, x1, y2, x2]
    b = box(minx=x_tile+x1, miny=y_tile+y1,
            maxx=x_tile+x2, maxy=y_tile+y2)
    return b


def apply_mask(mask_file, df):
    resize_dims = find_full_canvas_dims(df)

    mask = load_mask(mask_file, resize_dims)

    box_geoms = []
    for image, det_box in zip(df['image'], df['detection_boxes']):
        box_geoms.append(create_detection_geom(image, ast.literal_eval(det_box)))
    df['detection_box_geom'] = box_geoms

    masked_df = df[gpd.GeoSeries(df['detection_box_geom']).intersects(mask)]

    return masked_df


def mask_image(file_path, mask_file):

    image = Image.open(file_path)
    mask_polygon = load_mask(mask_file, image.size)
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(list(zip(*mask_polygon.exterior.xy)), fill=255, outline=20)
    black = Image.new("RGB", image.size, 0)
    result = Image.composite(image, black, mask)

    return result


def mask_and_tile(file_path, mask_file, out_dir, tile_height, tile_width):
    masked_image = mask_image(file_path, mask_file)
    tile_image(masked_image,
               tile_height=tile_height,
               tile_width=tile_width,
               out_dir=out_dir,
               out_extension='jpg',
               exclude_all_black_tiles=True)


if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='Takes a file path as input and tiles each .tif'
                                                 'file in that directory.')
    parser.add_argument('--in_dir', help='File path to the directory containing tifs to be tiled.')
    parser.add_argument('--in_file', help='File path to the directory containing tifs to be tiled.')
    parser.add_argument('--out_dir', help='File path to output directory.')
    parser.add_argument('--tile_height', type=int, help='Height in pixels of the resulting tiles')
    parser.add_argument('--tile_width', type=int, help='Width in pixels of the resulting tiles')
    args = parser.parse_args()

    if args.in_dir:
        tile_all_tifs(Path(args.in_dir),
                      args.tile_height,
                      args.tile_width,
                      Path(args.out_dir))
    elif args.in_file:
        tile_single_tif(Path(args.in_file),
                        args.tile_height,
                        args.tile_width,
                        Path(args.out_dir))

    tile_height = 3000
    tile_width = 3000
    #
    # # June 8
    # file_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2020_SNB/VALIDATION/SNB_08062020_74576x33620.tif'
    # mask_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/post_process_detections/input/2020_SNB/VALIDATION/20200608/span2_mask.csv'
    # out_dir = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/output/2020_SNB/MANUAL_COUNTS/20200608/'
    # mask_and_tile(Path(file_path), mask_file, Path(out_dir), tile_height, tile_width)
    #
    # # June 17
    # file_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2020_SNB/VALIDATION/SNB_17062020_71280x35076.tif'
    # mask_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/post_process_detections/input/2020_SNB/VALIDATION/20200617/span2_mask.csv'
    # out_dir = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/output/2020_SNB/MANUAL_COUNTS/20200617/'
    # mask_and_tile(Path(file_path), mask_file, Path(out_dir), tile_height, tile_width)
    #
    # # June 26
    # file_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2020_SNB/VALIDATION/SNB_26062020_71316x34752.tif'
    # mask_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/post_process_detections/input/2020_SNB/VALIDATION/20200626/span2_mask.csv'
    # out_dir = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/output/2020_SNB/MANUAL_COUNTS/20200626/'
    # mask_and_tile(Path(file_path), mask_file, Path(out_dir), tile_height, tile_width)
    #
    # # July 3
    # file_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2020_SNB/VALIDATION/SNB_03072020.tif'
    # mask_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/post_process_detections/input/2020_SNB/VALIDATION/20200703/span2_mask.csv'
    # out_dir = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/output/2020_SNB/MANUAL_COUNTS/20200703/'
    # mask_and_tile(Path(file_path), mask_file, Path(out_dir), tile_height, tile_width)
    #
    # # Aug 3
    # file_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2020_SNB/VALIDATION/SNB_03082020.tif'
    # mask_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/post_process_detections/input/2020_SNB/VALIDATION/20200803/span2_mask.csv'
    # out_dir = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/output/2020_SNB/MANUAL_COUNTS/20200803/'
    # mask_and_tile(Path(file_path), mask_file, Path(out_dir), tile_height, tile_width)
    #
    # # Sept 9
    # file_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2020_SNB/VALIDATION/SNB_09092020.tif'
    # mask_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/post_process_detections/input/2020_SNB/VALIDATION/20200909/span2_mask.csv'
    # out_dir = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/output/2020_SNB/MANUAL_COUNTS/20200909/'
    # mask_and_tile(Path(file_path), mask_file, Path(out_dir), tile_height, tile_width)
    #
    # # Sept 19
    # file_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2020_SNB/VALIDATION/SNB_18092020.tif'
    # mask_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/post_process_detections/input/2020_SNB/VALIDATION/20200918/span2_mask.csv'
    # out_dir = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/output/2020_SNB/MANUAL_COUNTS/20200918/'
    # mask_and_tile(Path(file_path), mask_file, Path(out_dir), tile_height, tile_width)
    #
    # # June 22
    # file_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2020_SNB/VALIDATION/SNB_22062020_70688x35540.tif'
    # mask_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/post_process_detections/input/2020_SNB/VALIDATION/20200622/span2_mask.csv'
    # out_dir = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/output/2020_SNB/MANUAL_COUNTS/20200622/'
    # mask_and_tile(Path(file_path), mask_file, Path(out_dir), tile_height, tile_width)

    # 2021
    # June 9
    file_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2021_SNB/VALIDATION/2021-06-09 SNB Panorama 13.tif'
    mask_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/post_process_detections/input/2021_SNB/TEST/SNB_13_20210609/span2_bridge_mask.csv'
    out_dir = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/output/2021_SNB/MANUAL_COUNTS/20210609/'
    mask_and_tile(Path(file_path), mask_file, Path(out_dir), tile_height, tile_width)

    # June 21
    file_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2021_SNB/VALIDATION/2021-06-21 SNB Panorama 18.tif'
    mask_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/post_process_detections/input/2021_SNB/TEST/SNB_18_20210621/span2_bridge_mask.csv'
    out_dir = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/output/2021_SNB/MANUAL_COUNTS/20210621/'
    mask_and_tile(Path(file_path), mask_file, Path(out_dir), tile_height, tile_width)

    # July 5
    file_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2021_SNB/VALIDATION/2021-07-05 SNB Panorama 23.tif'
    mask_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/post_process_detections/input/2021_SNB/TEST/SNB_23_20210705/span2_bridge_mask.csv'
    out_dir = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/output/2021_SNB/MANUAL_COUNTS/20210705/'
    mask_and_tile(Path(file_path), mask_file, Path(out_dir), tile_height, tile_width)

    # July 28
    file_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2021_SNB/VALIDATION/2021-07-28 SNB Panorama 33.tif'
    mask_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/post_process_detections/input/2021_SNB/TEST/SNB_33_20210728/span2_bridge_mask.csv'
    out_dir = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/output/2021_SNB/MANUAL_COUNTS/20210728/'
    mask_and_tile(Path(file_path), mask_file, Path(out_dir), tile_height, tile_width)

    # August 9
    file_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2021_SNB/VALIDATION/2021-08-09 SNB Panorama 36.tif'
    mask_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/post_process_detections/input/2021_SNB/TEST/SNB_36_20210809/span2_bridge_mask.csv'
    out_dir = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/output/2021_SNB/MANUAL_COUNTS/20210809/'
    mask_and_tile(Path(file_path), mask_file, Path(out_dir), tile_height, tile_width)

    # August 17
    file_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2021_SNB/VALIDATION/2021-08-17 SNB Panorama 38.tif'
    mask_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/post_process_detections/input/2021_SNB/TEST/SNB_38_20210817/span2_bridge_mask.csv'
    out_dir = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/output/2021_SNB/MANUAL_COUNTS/20210817/'
    mask_and_tile(Path(file_path), mask_file, Path(out_dir), tile_height, tile_width)

    # September 10
    file_path = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2021_SNB/VALIDATION/SNB_39_01092021.tif'
    mask_file = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/post_process_detections/input/2021_SNB/TEST/SNB_40_10092021/span2_bridge_mask.csv'
    out_dir = '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/tile_tifs/output/2021_SNB/MANUAL_COUNTS/20210910/'
    mask_and_tile(Path(file_path), mask_file, Path(out_dir), tile_height, tile_width)

