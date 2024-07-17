"""
e.g.
"""
import matplotlib.pyplot as plt
import pandas as pd
from PIL import ImageDraw, ImageFont, Image, ExifTags, __version__ as PILLOW_VERSION
from pathlib import Path
from shapely.geometry import box
import argparse
import json
import ast
import tqdm
import numpy as np
from functools import reduce
from dataclasses import dataclass

convert_version = lambda x : tuple(map(int, x.split(".")))
PILLOW_VERSION = convert_version(PILLOW_VERSION)
Image.MAX_IMAGE_PIXELS = 3000000000

@dataclass
class Sizes:
    """Class to parametrize drawing options (brush stroke) dependent on size of panorama image 
    These may vary before or after a reduce(). 
    """
    small: int
    medium: int
    large: int

brush_reduced = Sizes(1, 3, 5)
brush_raw = Sizes(5, 15, 75)
brush_size = brush_reduced

def filter_detections(detections, threshold_dict={}):
    for label, thresh in threshold_dict.items():
        detections = detections[(detections['detection_classes'] != label) |
                                ((detections['detection_classes'] == label) &
                                 (detections['detection_scores'] >= thresh))]
    return detections


def find_full_canvas_dims(df):
    tile_ys, tile_xs = zip(*df['image'].transform(lambda x: Path(x).name.split('.')[:2]))
    # TODO: This function is only used by the other unused function, can it be removed?
    tile_xs = [int(x) for x in tile_xs]
    tile_ys = [int(y) for y in tile_ys]
    return max(tile_xs) + 1, max(tile_ys) + 1


def create_detection_geom(detection_box, tile_width=1000, tile_height=1000, scale_factor=1):
    """ Create a Shapely bounding box for the `detection_box`
    
    To draw a `detection_box` box it must be like TensorFlow format and use entire pano coordinate.
    Inputs:
        detection_box (list): array of floating point values between 0 and 1, for coordinates [top, left, bottom, right] (TF format)
        tile_width (number): scaling factor to map 0-1 values into the actual tile width
        tile_height (number): scaling factor to map 0-1 values into the actual tile height
        scale_factor (number | [number,number] ): scaling factor to accommodate upscaling/rescaling due to reduce(). Default 1.

    Returns:
        box(shapely.geometry.box):
    """
    scale_factor_x, scale_factor_y = validate_scale(scale_factor)
    x1, y1, x2, y2 = detection_box  # format: [x1, y1, x2, y2]
    b = box(minx=(x1) * tile_width, miny=(y1) * tile_height,
            maxx=(x2) * tile_width, maxy=(y2) * tile_height)
    return b


def plot_mask_result(mask, df):
    # Plot Mask
    # TODO: This function is unused, can it be removed?
    plt.plot(*mask.exterior.xy, color='#FFD700', linewidth=2)  # Goldish

    # Plot Boxes
    box_geoms = []
    for image, det_box in zip(df['image'], df['detection_boxes']):
        box_geoms.append(create_detection_geom(ast.literal_eval(det_box)))

    for b in box_geoms:
        if b.intersects(mask):
            plt.plot(*b.exterior.xy, color='#87EC8D', alpha=0.5)  # Greenish

        else:
            plt.plot(*b.exterior.xy, color='#D84223', alpha=0.5)  # Redish

    # Show Image
    im = Image.open(img_file)
    x_max, y_max = find_full_canvas_dims(df)
    plt.imshow(im, extent=[0, 74.576, 33.620, 0], alpha=0.8)

    plt.xticks([])
    plt.yticks([])
    plt.savefig(
        'cormorants-nesting-scripts/object_detection_scripts/post_process_detections/output/diagrams/masking.png',
        dpi=600)


# From lost_to_tfrecord.py
def standardize_lost(image_directory, annotation_file):
    def get_img_size(img_name):
        image = Image.open(image_directory.joinpath(img_name))
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(image._getexif().items())

            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)

            image.save(image_directory.joinpath(img_name))
        except (AttributeError, KeyError, IndexError):
            # cases: image don't have getexif
            pass

        w, h = image.size
        image.close()

        return w, h

    # Read Annotations
    raw = pd.read_csv(annotation_file)
    annos = raw
    # annos = raw[~pd.isna(raw['anno.anno_task_id'])]   # Remove instances with no annotations

    # Filename
    annos['filename'] = annos['img.img_path'].apply(lambda x: Path(x).name)

    # Image Size
    annos['width'], annos['height'] = list(zip(*annos['filename'].apply(get_img_size)))

    # Class
    label_lists = annos['anno.lbl.name'].apply(lambda x: ast.literal_eval(x))
    label_lists.apply(lambda x: x[0] if len(x)>0 else None)
    annos['class'] = label_lists.apply(lambda x: x[0] if len(x)>0 else None)

    # Bounding Box Coordinates
    coord_list = [get_bbox_coords(d, w, h) for d, w, h in zip(annos['anno.data'], annos['width'], annos['height'])]
    annos['xmin'], annos['ymin'], annos['xmax'], annos['ymax'] = list(zip(*coord_list))

    nicely_named = annos[['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']]

    return nicely_named


def get_bbox_coords(raw_anno, w, h):
    if pd.isna(raw_anno):
        xmin = -1
        ymin = -1
        xmax = -1
        ymax = -1
    else:
        anno_dict = ast.literal_eval(raw_anno)
        xmin = (anno_dict['x']-anno_dict['w']/2) * w
        ymin = (anno_dict['y']-anno_dict['h']/2) * h
        xmax = (anno_dict['x']+anno_dict['w']/2) * w
        ymax = (anno_dict['y']+anno_dict['h']/2) * h

    return xmin, ymin, xmax, ymax


def standardize_lost(image_directory, annotation_file):
    def get_img_size(img_name):
        image = Image.open(image_directory.joinpath(img_name))
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(image._getexif().items())

            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)

            image.save(image_directory.joinpath(img_name))
        except (AttributeError, KeyError, IndexError):
            # cases: image don't have getexif
            pass

        w, h = image.size
        image.close()

        return w, h

    # Read Annotations
    raw = pd.read_csv(annotation_file)
    annos = raw
    # annos = raw[~pd.isna(raw['anno.anno_task_id'])]   # Remove instances with no annotations

    # Filename
    annos['filename'] = annos['img.img_path'].apply(lambda x: Path(x).name)

    # Image Size
    annos['width'], annos['height'] = list(zip(*annos['filename'].apply(get_img_size)))

    # Class
    label_lists = annos['anno.lbl.name'].apply(lambda x: ast.literal_eval(x))
    label_lists.apply(lambda x: x[0] if len(x)>0 else None)
    annos['class'] = label_lists.apply(lambda x: x[0] if len(x)>0 else None)

    # Bounding Box Coordinates
    coord_list = [get_bbox_coords(d, w, h) for d, w, h in zip(annos['anno.data'], annos['width'], annos['height'])]
    annos['xmin'], annos['ymin'], annos['xmax'], annos['ymax'] = list(zip(*coord_list))

    nicely_named = annos[['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']]

    return nicely_named


def draw_mask(draw, mask_file):
    df = pd.read_csv(mask_file)
    im_w, im_h = draw.im.size
    mask_points = [(im_w*d['x'], im_h*d['y']) for d in ast.literal_eval(df['anno.data'].iloc[0])]
    draw.polygon(mask_points, outline='orange', width=brush_size.large) #350 for super large
    return draw


def draw_tiles(draw, tile_height, tile_width):
    width, height = draw.im.size
    for x in range(int(width // tile_width) + 1 ):
        draw.line([(x*tile_width, 0), (x*tile_width, height)], fill='gainsboro', width=brush_size.small)

    for y in range(int(height // tile_height) + 1):
        draw.line([(0, y*tile_height), (width, y*tile_height)], fill='gainsboro', width=brush_size.small)
    
    draw.rectangle([0, 0, width, height], outline='gainsboro', width=brush_size.small)

    return draw


def draw_ground_truth_annotations(draw, ground_truth_file, tile_directory, tile_size=3000, rescale_factor=1):
    
    rescale_factor_x, rescale_factor_y = validate_scale(rescale_factor)

    annotations = pd.read_csv(ground_truth_file).dropna(subset=['anno.data'])
    font = ImageFont.load_default(size=12) if PILLOW_VERSION >= convert_version('10.1.0') else ImageFont.load_default()
    count = 0
    for i, anno, i_file in zip(annotations['anno.idx'], annotations['anno.data'], annotations['img.img_path']):
        tile_name = Path(i_file).name
        tile = Image.open(tile_directory.joinpath(tile_name))
        tile_y, tile_x, _ = tile_name.split('.')
        tile_w, tile_h = tile.size
        tile_x0, tile_y0, tile_x1, tile_y1 = get_bbox_coords(anno, tile_w, tile_h)
        x0 = (tile_x0 + (int(tile_x)*tile_size)) / rescale_factor_x
        y0 = (tile_y0 + (int(tile_y)*tile_size)) / rescale_factor_y
        x1 = (tile_x1 + (int(tile_x)*tile_size)) / rescale_factor_x
        y1 = (tile_y1 + (int(tile_y)*tile_size)) / rescale_factor_y
        draw.rectangle([x0, y0, x1, y1], outline='gold', width=brush_size.medium)
        draw.text((np.mean([x0, x1]), np.mean([y0, y1])), str(int(i)), fill='DeepPink', font=font, anchor='mm')

    return draw

def validate_scale(scale_factor):
    """ Validate argument inputs for 1D or 2D 'scale' or 'rescale' -ing.

        TODO: Can this function be replaced with appropriate type hint? Take into consideration python 3.7 support due to object_detection TF package in the pip environment
        scale_factor (_type_): int | float | numeric | [numeric, numeric] | (numeric, numeric)

    Returns:
        numeric, numeric: pair of scale factors for _x and _y coordinates
            if a single number is provided in `scale_factor`, then the same scale is used for _x and _y.
            if a pair of numbers are provided in `scale_factor`, then they are returned for _x and _y, respectively.
            Default 1,1.
    """
    
    if scale_factor == None:
        rescale_factor_x, rescale_factor_y = 1, 1

    elif isinstance(scale_factor, int) or isinstance(scale_factor, float):
        rescale_factor_x = scale_factor
        rescale_factor_y = scale_factor
    
    elif isinstance(scale_factor, tuple) or isinstance(scale_factor, list):
        assert len(scale_factor) == 2, "Attempting to scale along 3 dimensions. Only scaling in 2 dimensions is supported."
        
        rescale_factor_x, rescale_factor_y = scale_factor[0], scale_factor[1]
        
        assert (isinstance(rescale_factor_x, int) or isinstance(rescale_factor_x, float)) and (isinstance(rescale_factor_y, int) or isinstance(rescale_factor_y, float)), f"Non numeric scale factors ({scale_factor}) are provided. Must be of type int or float."
    
    else:
        rescale_factor_x, rescale_factor_y = 1, 1
        print(f"Warning: scale factor ({scale_factor}) could not be validated, setting rescaling to 1")

    return rescale_factor_x, rescale_factor_y


def main(rescale_factor=4):
    if detections_file is not None:
        detections = pd.read_csv(detections_file)
        detections = detections.reset_index() # ensure RangeIndex is converted to list index to produce IDs
        
        assert detections['detection_classes'].hasnans == False, f'detections_file={detections_file} must not contain null classes' + '\n' + 'use output of post_process_detections'
        
        detections = filter_detections(detections, threshold_dict)

    print("Reading in Image")
    im = Image.open(img_file)
    width_actual, height_actual = im.size

    print("Reducing Image")
    im = im.reduce(factor=rescale_factor)
    width_reduced, height_reduced = im.size
    
    width_scale = width_actual / width_reduced
    height_scale = height_actual / height_reduced
    upscale_factor = width_scale, height_scale # NOTE: due to floating point multiplication and division, fp rounding errors may occur e.g. 1e-11

    print(f"New image size is: {width_reduced, height_reduced}")
    print(f"""
        {width_actual=}, {height_actual=}
        {width_scale=}, {height_scale=}
        {width_reduced=}, {height_reduced=}
        scaled_back={width_reduced * width_scale}, {height_reduced * height_scale}
        {upscale_factor=}
        """)

    print("Collect Boxes")
    if detections_file is not None:
        box_geoms = []
        box_labels = []
        for image, det_box, label, ind in zip(detections['image'], detections['detection_boxes'],
                                         detections['detection_classes'], detections['index']):
            box_geoms.append(
                create_detection_geom(ast.literal_eval(det_box), tile_width=tile_size/height_scale,
                                      tile_height=tile_size/width_scale))
            box_labels.append((ind, label))    

    print("Loading Canvas")
    draw = ImageDraw.Draw(im)

    print("Draw Tile Borders")
    draw = draw_tiles(draw, tile_size/height_scale, tile_size/width_scale)

    print("Draw Mask")
    if mask_file and mask_file.is_file() and mask_file.exists(): 
        draw = draw_mask(draw, mask_file)

    print("Drawing Boxes")
   
    font = ImageFont.load_default(size=12) if PILLOW_VERSION >= convert_version('10.1.0') else ImageFont.load_default()

    if detections_file is not None:
        for b, detect in tqdm.tqdm(zip(box_geoms, box_labels), total=len(box_labels)):
            idx, lbl = detect
            if lbl == 0:
                color = '#90EE90'
            elif lbl == 1:
                color = '#fc8d59'
                
            coords = list(zip(*b.exterior.xy))
            draw_box(draw, color, coords)  

            horizontal_alignment = 'right'
            text_str = f"detection_id: {idx:.0f}"

            draw_text(draw, coords, text_str=text_str, align=horizontal_alignment, font=font, outline=False)

    #print("Draw Ground truth Annotations")
    if ground_truth_file and tile_directory and ground_truth_file.is_file() and tile_directory.exists():
        print("Draw Ground truth Annotations")
        draw = draw_ground_truth_annotations(ImageDraw.Draw(im), ground_truth_file, tile_directory,
                                             tile_size = anno_tile_size, rescale_factor=1)
    else:
        print("Skipping Ground truth Annotations")

    print("Saving Result")
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    im.save(out_file)


def draw_text(draw:ImageDraw, coords:list[tuple], text_str:str, align:str, font:ImageFont, outline=True):
    """Write `text` inside at bounding box attached to the upper left corner of the existing bbox at `coords`.

    Args:
        draw (ImageDraw): _description_
        coords (list[tuple]): Sequence of coordinate pairs for the bbox or polygon. The smallest corner (upper left; 0,0) will be used to anchor the text bbox and text
        text_str (str): Text to write
        align (str): Same as the 'align' parameter in ImageDraw.text. Corresponds to horizontal alignment
        font (ImageFont): _description_
        outline (bool): Whether to draw the enclosing bounding box or just the text.
    """
    
    bbox_x0, bbox_y0 = reduce(min, map(lambda x: x[0], coords)), reduce(min, map(lambda x: x[1], coords)) # left ascender corner of bbox

    text_box = font.getbbox(text_str) # text_box = x1, y1, x2, y2
    # Coordinates are calculated manually (using the upper left corner, 0,0 as reference origin)
    # TODO: Calculate text box coordinates programatically using shapely geometric objects
    text_height = text_box[3] - text_box[1] 
    
    text_box = draw.textbbox((bbox_x0, bbox_y0-text_height), 
                            #  adjust the top-left coordinate by text_height
                            text_str,
                            font=font,
                            align=align,
                            anchor='la',
                            # font_size=32,  # Added in version 10.1.0.
                            )
    
    if outline:
        draw.rectangle(text_box, outline='red', width=brush_size.medium)
    draw.text((text_box[0], text_box[1]), text_str, fill='red', font=font, anchor='la') # left ascender corner of text box

def draw_box(draw:ImageDraw, color, coords):
    if PILLOW_VERSION >= convert_version('9.0'):
        draw.polygon(coords, outline=color, width=brush_size.medium)
    else:
        draw.line(coords, fill=color, width=brush_size.medium)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_file', type=str, help='File path to the input file panorama on which to draw detections.')

    parser.add_argument('--detections_file', type=str, help='File path to the detections predicted by model inference.')
    parser.add_argument('--mask_file', type=str, required=False, help='File path to the CSV file containing the mask which'
                                       'corresponds to the TIF(s) being tiled. There should be'
                                       'a 1:1 correspondence between the TIF and mask names.')
    
    parser.add_argument('--tile_size', type=int, default=1000, required=False, help='Pixels for square (detection) tiles')
    parser.add_argument('--rescale_factor', type=int, default=4, required=False, help='Compression factor for output image and size.')
    parser.add_argument('--threshold_dict', type=json.loads, default='{"0.0": 0.2, "1.0": 0.2}', help='cutoff scores for each class {0.0: 0.2, 1.0: 0.2} # 0: Cormorants, 1: Nest')
    
    parser.add_argument('--anno_tile_size', type=int, default=1000, required=False, help='Pixels for square (annotation) tiles.')
    parser.add_argument('--ground_truth_file', type=str, required=False, help='File path to ground truth annotations.')
    parser.add_argument('--tile_directory', type=str, required=False, help='Path to the tile_directory containing the tiles corresponding to the ground truth annotations.')

    parser.add_argument('--out_file', type=str, help='File path to img_file with boxes drawn from model predictions.')
    args = parser.parse_args()

    img_file = Path(args.img_file)
    detections_file = Path(args.detections_file)
    mask_file = Path(args.mask_file) if args.mask_file else args.mask_file

    # TODO: check for duplication/overlap in `draw_ground_truth_annotations`
    tile_size = args.tile_size
    anno_tile_size = args.anno_tile_size
    rescale_factor = args.rescale_factor
    threshold_dict = args.threshold_dict
    threshold_dict = { float(k): float(v) for k,v in threshold_dict.items()}

    ground_truth_file = args.ground_truth_file
    tile_directory = args.tile_directory
    try:
        ground_truth_file = Path(args.ground_truth_file) # if args.ground_truth_file else args.ground_truth_file
        tile_directory = Path(args.tile_directory) # if args.tile_directory else args.tile_directory
    except TypeError as err: #expected str, bytes or os.PathLike object, not NoneType
        print(f"ground_truth_file={ground_truth_file}, tile_directory={tile_directory}: {str(err)}")
        print("skipping drawing ground truth annotations")
    
    out_file = Path(args.out_file) / img_file.name
    
    print("DEBUG:")
    print(f"  img_file={img_file}", f"detections_file={detections_file}", f"mask_file={mask_file}", sep="\n  ", end='\n\n')
    print(f"  threshold_dict={threshold_dict}", f"tile_size={tile_size}", f"anno_tile_size={anno_tile_size}", f"rescale_factor={rescale_factor}", sep="\n  ", end='\n\n')
    print(f"  ground_truth_file={ground_truth_file}", f"tile_directory={tile_directory}", sep="\n  ", end='\n\n')
    print(f"  out_file={out_file}", sep="\n  ", end='\n\n')

    main()
