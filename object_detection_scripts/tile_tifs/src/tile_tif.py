"""
Based off
github.com/jillianderson8/cmpt733finalproject/blob/master/Code/3_tileAnnotate/src/tileAnnotate.py

e.g.
python3 tile_tif.py \
--in_dir ../input/2020_SNB/ \
--out_dir ../output/2020_SNB/ \
--tile_height 1000 \
--tile_width 1000

"""
import math
from PIL import Image
import argparse
import os
from pathlib import Path

Image.MAX_IMAGE_PIXELS = 3000000000


def tile_image(image, tile_height, tile_width, out_dir, out_extension):
    """
        Divide the given image into tiles
    """
    img_width, img_height = image.size

    # Create tile output directory
    out_dir.mkdir(exist_ok=True)

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

            try:
                # Save Tile Image
                file_name = f"{i}.{j}.{out_extension}"
                out_path = out_dir.joinpath(file_name)
                tile.convert('RGB').save(out_path)

            except Exception as e:
                print(e)


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


if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='Takes a file path as input and tiles each .tif'
                                                 'file in that directory.')
    parser.add_argument('--in_dir', help='File path to the directory containing tifs to be tiled.')
    parser.add_argument('--out_dir', help='File path to output directory.')
    parser.add_argument('--tile_height', type=int, help='Height in pixels of the resulting tiles')
    parser.add_argument('--tile_width', type=int, help='Width in pixels of the resulting tiles')
    args = parser.parse_args()

    tile_all_tifs(Path(args.in_dir),
                  args.tile_height,
                  args.tile_width,
                  Path(args.out_dir))