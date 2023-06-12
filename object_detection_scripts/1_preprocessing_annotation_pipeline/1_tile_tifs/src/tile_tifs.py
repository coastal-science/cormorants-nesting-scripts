"""
Source code for splitting a large TIF image up into smaller JPG tiles.
"""
import ast
import math
import shapely
import argparse
import pandas as pd
from shapely.geometry import Polygon
from PIL import Image, ImageFile, ImageDraw
from pathlib import Path

Image.MAX_IMAGE_PIXELS = 3000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


def tile_single_tif(tif_file, mask_file, tile_height, tile_width, out_directory):
    # Open Image
    image = Image.open(tif_file)

    # Mask Image
    if mask_file:
        image = _mask_image(tif_file, mask_file)

    tif_tile_dir = out_directory.joinpath(tif_file.stem)

    # Tile Image
    _tile_image(image,
                tile_height=tile_height,
                tile_width=tile_width,
                out_dir=tif_tile_dir,
                out_extension='jpg',
                exclude_all_black_tiles=bool(mask_file))


def tile_all_tifs(tif_directory, mask_file, tile_height, tile_width, out_directory):
    all_tifs = tif_directory.glob("*.tif")
    for tif_file in all_tifs:
        # Open Image
        image = Image.open(tif_file)

        # Mask Image
        if mask_file:
            image = _mask_image(tif_file, mask_file)

        # Tile Image
        tif_tile_dir = out_directory.joinpath(tif_file.stem)
        _tile_image(image,
                    tile_height=tile_height,
                    tile_width=tile_width,
                    out_dir=tif_tile_dir,
                    out_extension='jpg',
                    exclude_all_black_tiles=bool(mask_file))


def _tile_image(image, tile_height, tile_width, out_dir, out_extension, exclude_all_black_tiles=False):
    """
        Divide the given image into tiles
    """
    img_width, img_height = image.size

    # Create tile output tif_directory
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


def _load_mask(mask_file, tif_name, resize_dims):
    df = pd.read_csv(mask_file)
    tif_mask_mapping = {Path(t).name: m for t, m in zip(df['img.img_path'], df['anno.data'])}
    try:
        mask = tif_mask_mapping[tif_name]
        print(f"Loading mask for {df['img.img_path'].iloc[0]}")
    except KeyError:
        raise KeyError("Could not find a matching mask.")

    mask_points = [(d['x'], d['y']) for d in ast.literal_eval(mask)]
    resized_points = [(x*resize_dims[0], y*resize_dims[1]) for x, y in mask_points]
    mask_geom = shapely.geometry.Polygon(resized_points)

    return mask_geom


def _mask_image(tif_path, mask_file):

    image = Image.open(tif_path)
    mask_polygon = _load_mask(mask_file, tif_path.name, image.size)
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(list(zip(*mask_polygon.exterior.xy)), fill=255, outline=20)
    black = Image.new("RGB", image.size, 0)
    result = Image.composite(image, black, mask)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--tif_file', help='File path to the TIF to be tiled.')
    parser.add_argument('--mask_file', help='File path to the CSV file containing the mask which'
                                            'corresponds to the TIF(s) being tiled. There should be'
                                            'a 1:1 correspondence between the TIF and mask names.')
    parser.add_argument('--tif_dir', help='File path to the tif_directory containing TIFs to be '
                                          'tiled.')
    parser.add_argument('--out_dir', help='File path to tif_directory where the folder containing '
                                          'tiles will be placed.')
    parser.add_argument('--tile_height', type=int, help='Height in pixels of the resulting tiles')
    parser.add_argument('--tile_width', type=int, help='Width in pixels of the resulting tiles')
    args = parser.parse_args()

    if args.tif_dir:
        tile_all_tifs(tif_directory=Path(args.tif_dir),
                      mask_file=Path(args.mask_file) if args.mask_file else None,
                      tile_height=args.tile_height,
                      tile_width=args.tile_width,
                      out_directory=Path(args.out_dir))
    elif args.tif_file:
        tile_single_tif(tif_file=Path(args.tif_file),
                        mask_file=Path(args.mask_file) if args.mask_file else None,
                        tile_height=args.tile_height,
                        tile_width=args.tile_width,
                        out_directory=Path(args.out_dir))