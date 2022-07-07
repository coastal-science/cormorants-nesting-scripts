import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import argparse
from pathlib import Path
from shapely.geometry import Polygon, box
import ast

Image.MAX_IMAGE_PIXELS = 3000000000


def find_full_canvas_dims(df):
    tile_ys, tile_xs = zip(*df['image'].transform(lambda x: Path(x).name.split('.')[:2]))
    tile_xs = [int(x) for x in tile_xs]
    tile_ys = [int(y) for y in tile_ys]
    return max(tile_xs) + 1, max(tile_ys) + 1


def create_detection_geom(tile_name, detection_box):
    y_tile, x_tile = Path(tile_name).name.split('.')[:2]
    y_tile = int(y_tile)
    x_tile = int(x_tile)

    y1, x1, y2, x2 = detection_box  # format: [y1, x1, y2, x2]
    b = box(minx=x_tile+x1, miny=y_tile+y1,
            maxx=x_tile+x2, maxy=y_tile+y2)
    return b


def plot_mask_result(mask, df):
    # Plot Mask
    plt.plot(*mask.exterior.xy, color='#FFD700', linewidth=2)        # Goldish

    # Plot Boxes
    box_geoms = []
    for image, det_box in zip(df['image'], df['detection_boxes']):
        box_geoms.append(create_detection_geom(image, ast.literal_eval(det_box)))

    for b in box_geoms:
        if b.intersects(mask):
            plt.plot(*b.exterior.xy, color='#87EC8D', alpha=0.5)    # Greenish

        else:
            plt.plot(*b.exterior.xy, color='#D84223', alpha=0.5)   # Redish

    # Show Image
    im = Image.open(img_file)
    x_max, y_max = find_full_canvas_dims(df)
    plt.imshow(im, extent=[0, 74.576, 33.620, 0], alpha=0.8)

    plt.xticks([])
    plt.yticks([])
    plt.savefig('cormorants-nesting-scripts/object_detection_scripts/post_process_detections/output/diagrams/masking.png', dpi=600)


def main(detections_file, rescale_factor=2):
    detections = pd.read_csv(detections_file)

    # Plot Boxes
    box_geoms = []
    box_labels = []
    for image, det_box, label in zip(detections['image'], detections['detection_boxes'], detections['detection_classes']):
        box_geoms.append(create_detection_geom(image, ast.literal_eval(det_box)))
        box_labels.append(label)

    for b, lbl in zip(box_geoms, box_labels):
        if lbl == 0:
            color = '#90EE90'
        elif lbl == 1:
            color = '#EE82EE'
        plt.plot(*b.exterior.xy, color=color, alpha=0.8, linewidth=3)  # Plum

    # Add Image
    # Show Image
    fullsize_im = Image.open(img_file)
    im = fullsize_im.reduce(factor=rescale_factor)
    width, height = im.size
    plt.imshow(im, extent=[0, width/tile_size*rescale_factor, height/tile_size*rescale_factor, 0], alpha=0.8)    # [left, right, bottom, top]

    # Save Image
    out_dir = Path(out_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_file', help='.', type=str)
    parser.add_argument('--detections_file', help='.', type=str)
    parser.add_argument('--mask_file', help='.', type=str)
    parser.add_argument('--out_file', help='.', type=str)
    args = parser.parse_args()

    # img_file = 'cormorants-nesting-scripts/object_detection_scripts/resize_image/output/2022/SNB_Span1/06_June/SNBS1_20220604_REDUCED_10.jpg'
    img_file = 'cormorants-nesting-scripts/object_detection_scripts/tile_tifs/input/2022_SNB_Span1/06_June/June 04, Span 1 Panorama.tif'
    detections_file = 'cormorants-nesting-scripts/object_detection_scripts/post_process_detections/output/2022/SNB_Span1/06_June/SNB_S1_20220604_500_snb3_v1_detections_pp3.csv'
    out_file = 'cormorants-nesting-scripts/draw_final_detections/output/2022/SNB_Span1/06_June/SNBS1_20220604_500.png'
    tile_size = 500
    # main(args.detections_file, args.mask_file, args.out_file)