import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import pickle
from matplotlib.patches import Rectangle
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
import argparse
from pathlib import Path
import math

Image.MAX_IMAGE_PIXELS = 3000000000


def create_box_geom(bbox):
    bbox_x, bbox_y, bbox_w, bbox_h = bbox

    b = box(minx=bbox_x,
            miny=bbox_y,
            maxx=bbox_x + bbox_w,
            maxy=bbox_y + bbox_h)

    return b


def get_all_boxes(coco_eval, false_positives):
    # Get Detections
    dt_df = pd.DataFrame(coco_eval.cocoDt.anns.values())
    dt_df = dt_df[['id', 'image_id', 'category_id', 'bbox', 'score']]
    all_false_positive_ids = [fp['id'] for fp in false_positives]
    dt_df['box_type'] = dt_df['id'].apply(
        lambda x: 'false_positive_detection' if x in all_false_positive_ids else 'true_positive_detection')

    # Get Annotations
    gt_df = pd.DataFrame(coco_eval.cocoGt.anns.values())
    gt_df = gt_df[['id', 'image_id', 'category_id', 'bbox']]
    gt_df['box_type'] = 'ground_truth_annotation'
    gt_df['id'] = gt_df['id'].astype(int)

    df = pd.concat([dt_df, gt_df])

    # Convert Bbox to shapely
    df['bbox_geom'] = df['bbox'].apply(create_box_geom)

    return df


def get_intersecting_boxes(all_boxes, area_of_interest, img_id):
    image_bboxs = all_boxes[all_boxes['image_id'] == img_id]
    intersecting_boxes = image_bboxs[
        gpd.GeoSeries(image_bboxs['bbox_geom']).intersects(area_of_interest)]
    return intersecting_boxes


def get_false_positives(results, coco_eval):
    # Find all the False Positive Errors -- Detections that weren't matched
    false_positives = []
    for res in results:
        for d in res['dtIds']:
            if d not in res['gtMatches']:
                false_positives.append(coco_eval.cocoDt.anns[d])

    return false_positives


def get_false_negatives(results, coco_eval):
    # Find all the False Negative Errors -- Annotations that weren't matched
    false_negatives = []
    for res in results:
        for g in res['gtIds']:
            if g not in res['dtMatches']:
                false_negatives.append(coco_eval.cocoGt.anns[g])
    return false_negatives


def get_draw_params(ann_type, category):
    color = 'black'

    if ann_type == 'false_positive_detection':
        if category == 0:  # Cormorant
            color = 'dodgerblue'
        elif category == 1:  # Nest
            color = 'crimson'
    elif ann_type == 'true_positive_detection':
        if category == 0:  # Cormorant
            color = 'lightgreen'
        elif category == 1:  # Nest
            color = 'lightsalmon'
    elif ann_type == 'ground_truth_annotation':
        if category == 0:  # Cormorant
            color = 'gold'
        elif category == 1:  # Nest
            color = 'pink'

    if 'detection' in ann_type:
        horizontal_alignment = 'left'
    else:
        horizontal_alignment = 'right'

    return color, horizontal_alignment


def point_converter(crop_geom, x, y):
    crop_x0, crop_y0, _, _ = crop_geom.bounds
    new_x = x - crop_x0
    new_y = y - crop_y0
    return new_x, new_y


def draw_bbox(bbox_row, crop_geom, axes, focus_box_id):
    bbox_center_x, bbox_center_y, bbox_w, bbox_h = bbox_row['bbox']
    bbox_x0, bbox_y0 = point_converter(crop_geom, bbox_center_x,
                                       bbox_center_y)


    color, horizontal_alignment = get_draw_params(bbox_row['box_type'], bbox_row['category_id'])
    linestyle = 'dashed' if bbox_row['id'] == focus_box_id else 'solid'
    text_format = f"{bbox_row['id']:.0f}--{bbox_row['score']:.2f}" if 'detection' in bbox_row[
        'box_type'] else f"{bbox_row['id']:.0f}"

    axes.add_patch(Rectangle((bbox_x0, bbox_y0), bbox_w, bbox_h,
                             linewidth=2, edgecolor=color,
                             facecolor='none', alpha=0.8, linestyle=linestyle))

    axes.text(x=min(1000, max(0, bbox_x0) + 6),
              y=min(1000, max(0, bbox_y0) - 3),
              s=text_format, fontsize=6, color=color, alpha=0.8,
              horizontalalignment=horizontal_alignment)


def get_crop(img, x, y, buffer):
    x0 = x - buffer
    y0 = y - buffer
    x1 = x + buffer
    y1 = y + buffer

    cropped_im = img.crop(box=(x0, y0, x1, y1))
    image_geom = box(x0, y0, x1, y1)

    return cropped_im, image_geom


def draw_tiles(plt_axes, crop_geom, tile_size):
    tile_width, tile_height = tile_size

    x0, y0, x1, y1 = crop_geom.bounds

    x_tile_edges = range(
        math.ceil(x1//tile_width * tile_width),
        math.floor(x0//tile_width * tile_width),
        -round(tile_width)
    )
    for edge in x_tile_edges:
        converted_x_edge, _ = point_converter(crop_geom, edge, 0)
        plt_axes.axvline(x=converted_x_edge, linewidth=1, color='gainsboro', alpha=0.6)

    y_tile_edges = range(
        math.ceil(y1 // tile_height * tile_height),
        math.floor(y0 // tile_height * tile_height),
        -round(tile_height)
    )
    for edge in y_tile_edges:
        _, converted_y_edge = point_converter(crop_geom, 0, edge)
        plt_axes.axhline(y=converted_y_edge, linewidth=1, color='gainsboro', alpha=0.6)


def generate_error_validation_images(img, img_id, errors, all_boxes, out_dir,
                                     tile_size=(1000, 1000)):
    for err in tqdm(errors):
        bbox_x, bbox_y, bbox_w, bbox_h = err['bbox']
        crop_img, crop_geom = get_crop(img, bbox_x, bbox_y, buffer=500)
        boxes_to_draw = get_intersecting_boxes(all_boxes, crop_geom, img_id)

        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(crop_img)
        axarr[1].imshow(crop_img)

        boxes_to_draw.apply(draw_bbox, axis=1, args=(crop_geom, axarr[0], err['id']))

        draw_tiles(axarr[0], crop_geom, tile_size)
        # draw_tiles(axarr[1], crop_geom, tile_size)

        axarr[0].axis('off')
        axarr[1].axis('off')

        if 'score' in err:
            file_path = f"{err['score']:.2f}_{err['id']:.0f}.png"
        else:
            file_path = f"{err['id']:.0f}.png"
        plt.savefig(Path(out_dir).joinpath(file_path), dpi=800, bbox_inches='tight')
        plt.close()


def create_error_evaluation_images(coco_eval_file, img_map, out_dir,
                                   false_positives=True, false_negatives=True):
    with open(coco_eval_file, "rb") as f:
        coco_eval = pickle.load(f)

    results = [c for c in coco_eval.evalImgs if c['aRng'] == coco_eval.params.areaRng[0]]

    false_positive_boxes = get_false_positives(results, coco_eval=coco_eval)
    false_negative_boxes = get_false_negatives(results, coco_eval=coco_eval)
    all_boxes = get_all_boxes(coco_eval, false_positive_boxes)

    for img_id in img_map:
        img = Image.open(img_map[img_id])

        if false_positives:
            print(f"\nGenerating False Positive Images for {img_id}\n")
            fp_out_dir = Path(out_dir).joinpath("false_positives").joinpath(f"{img_id}")
            fp_out_dir.mkdir(exist_ok=True, parents=True)
            img_false_positives = [fp for fp in false_positive_boxes if fp['image_id'] == img_id]
            generate_error_validation_images(img, img_id, img_false_positives, all_boxes,
                                             fp_out_dir)

        if false_negatives:
            print(f"\nGenerating False Negative Images for {img_id}\n")
            fn_out_dir = Path(out_dir).joinpath("false_negatives").joinpath(f"{img_id}")
            fn_out_dir.mkdir(exist_ok=True, parents=True)
            img_false_negatives = [fn for fn in false_negative_boxes if fn['image_id'] == img_id]
            generate_error_validation_images(img, img_id, img_false_negatives, all_boxes,
                                             fn_out_dir)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='')
    #
    # args = parser.parse_args()

    coco_eval_file_dir = Path("/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/4_comparing_manual_counts/calc_mAP/output")
    coco_eval_file = coco_eval_file_dir.joinpath("2020_val_eval_iou10.pkl")

    out_dir = "/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts/object_detection_scripts/4_comparing_manual_counts/draw_errors/output/2020/VALIDATION"

    # Draw the False Positive Errors
    img_map = {
        # 20200608: '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts-bkp/object_detection_scripts/1_preprocessing_annotation_pipeline/tile_tifs/input/2020_SNB/VALIDATION/SNB_08062020_74576x33620.tif',
        # 20200617: '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts-bkp/object_detection_scripts/1_preprocessing_annotation_pipeline/tile_tifs/input/2020_SNB/VALIDATION/SNB_17062020_71280x35076.tif',
        # 20200622: '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts-bkp/object_detection_scripts/1_preprocessing_annotation_pipeline/tile_tifs/input/2020_SNB/VALIDATION/SNB_22062020_70688x35540.tif',
        # 20200626: '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts-bkp/object_detection_scripts/1_preprocessing_annotation_pipeline/tile_tifs/input/2020_SNB/VALIDATION/SNB_26062020_71316x34752.tif',
        # 20200703: '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts-bkp/object_detection_scripts/1_preprocessing_annotation_pipeline/tile_tifs/input/2020_SNB/VALIDATION/SNB_03072020.tif',
        # 20200803: '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts-bkp/object_detection_scripts/1_preprocessing_annotation_pipeline/tile_tifs/input/2020_SNB/VALIDATION/SNB_03082020.tif',
        # 20200909: '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts-bkp/object_detection_scripts/1_preprocessing_annotation_pipeline/tile_tifs/input/2020_SNB/VALIDATION/SNB_09092020.tif',
        20200918: '/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts-bkp/object_detection_scripts/1_preprocessing_annotation_pipeline/tile_tifs/input/2020_SNB/VALIDATION/SNB_18092020.tif',
    }

    create_error_evaluation_images(coco_eval_file, img_map, out_dir)
#
#
#     img_id = 20200608
#     img_detections = [d for d in coco_eval.cocoDt.anns.values() if d['image_id'] == img_id]
#     nes_detections = [d for d in img_detections if d['category_id']==1]
#     len(img_detections)
#
#     results = [c for c in coco_eval.evalImgs if c['aRng'] == coco_eval.params.areaRng[0]]
#
#
#
# annotation_box = coco_eval.cocoGt.anns[18033]['bbox']
# detection_box = [d for d in nes_detections if d['id']==2693][0]['bbox']
#
# from matplotlib import pyplot as plt, patches
# fig = plt.figure()
# ax = fig.add_subplot(111)
# a_x, a_y = point_converter(crop_geom, annotation_box[0], annotation_box[1])
# d_x, d_y = point_converter(crop_geom, detection_box[0], detection_box[1])
#
# rectangle = patches.Rectangle((a_x, a_y), annotation_box[2], annotation_box[3], edgecolor='gold', fill=None, linewidth=2)
# ax.add_patch(rectangle)
#
# rectangle = patches.Rectangle((d_x, d_y), detection_box[2], detection_box[3], edgecolor='red', fill=None, linewidth=2)
# ax.add_patch(rectangle)
#
#
# plt.xlim([a_x-500, a_x+500])
# plt.ylim([a_y-500, a_y+500])
# plt.gca().invert_yaxis()