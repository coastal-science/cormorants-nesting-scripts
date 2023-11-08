import argparse
from pathlib import Path
import pandas as pd
import ast
import re
import pickle as pkl
import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from PIL import Image
import json

# Bounding Box Helpers
class BoundingBox:
    def __init__(self):
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0

        self.x_center = 0
        self.y_center = 0

        self.width = 0
        self.height = 0

    def load_four_corners(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

        self.width = x1 - x0
        self.height = y1 - y0

        self.x_center = (x1 + x0) / 2
        self.y_center = (y1 + y0) / 2

    def four_corners(self):
        return self.x0, self.y0, self.x1, self.y1

    def geom(self):
        pass


def resize_annotation(row, tile_directory, rescale_factor):
    raw_anno = ast.literal_eval(row['detection_boxes'])
    raw_path = row['img.img_path']
    img_date = re.search("[0-9]{8}", raw_path).group()
    tile_name = Path(raw_path).name

    tile = Image.open(tile_directory.joinpath(img_date).joinpath(tile_name))
    tile_y, tile_x, _ = tile_name.split('.')
    tile_w, tile_h = tile.size

    tile_x0, tile_y0, tile_x1, tile_y1 = raw_anno
    x0 = tile_x0*tile_w / rescale_factor
    y0 = tile_y0*tile_h / rescale_factor
    x1 = tile_x1*tile_w / rescale_factor
    y1 = tile_y1*tile_h / rescale_factor

    return x0, y0, x1, y1


def resize_detection(row, tile_size=(1000, 1000), rescale_factor=1):
    raw_detection = ast.literal_eval(row['detection_boxes'])
    raw_path = row['image']
    tile_name = Path(raw_path).name
    tile_y, tile_x, _ = tile_name.split('.')
    tile_w, tile_h = tile_size
    tile_x0, tile_y0, tile_x1, tile_y1 = raw_detection
    tile_x0 = tile_x0-int(tile_x)
    tile_y0 = tile_y0-int(tile_y)
    tile_x1 = tile_x1-int(tile_x)
    tile_y1 = tile_y1-int(tile_y)

    x0 = (tile_x0*tile_w + (int(tile_x)*tile_w)) / rescale_factor
    y0 = (tile_y0*tile_h + (int(tile_y)*tile_h)) / rescale_factor
    x1 = (tile_x1*tile_w + (int(tile_x)*tile_w)) / rescale_factor
    y1 = (tile_y1*tile_h + (int(tile_y)*tile_h)) / rescale_factor

    return x0, y0, x1, y1


def convert_annotations(dataframes):
    all_annotations = pd.concat(dataframes).dropna(subset=['detection_boxes']).apply(annotation_to_json,
                                                                               axis=1)

    return all_annotations


def annotation_to_json(row):
    # Get X, Y, X Y box & Resize it
    resized_anno = resize_annotation(row, tile_directory, rescale_factor=1)
    # Convert it to COCO box
    bbox = x1_y1_x2_y2_conversion(resized_anno)

    img_date = re.search("[0-9]{8}", row['img.img_path']).group()
    annotation = {
        'id': row['anno.idx'],
        'image_id': int(img_date),
        'category_id': label_map[ast.literal_eval(row['anno.lbl.name'])[0]],
        'area': bbox[2] * bbox[3],
        'bbox': bbox,
        'iscrowd': 0
    }

    return annotation


def detection_to_json(row):
    # Get X, Y, X Y box & Resize it
    resized_detection = resize_detection(row, tile_size=(1000,1000), rescale_factor=1)

    # Convert it to COCO box
    bbox = x1_y1_x2_y2_conversion(resized_detection)

    img_date = re.search("[0-9]{8}", row['image']).group()
    result = {
        "image_id": int(img_date),
        "category_id": row['detection_classes'],
        "bbox": bbox,
        "score": row['detection_scores']
    }

    return result


def convert_detections(dataframes):
    all_detections = pd.concat(dataframes)
    # all_detections = all_detections[all_detections['detection_scores'] >= 0.2]
    all_detections = all_detections.apply(detection_to_json, axis=1)
    return all_detections


def x1_y1_x2_y2_conversion(bbox):
    x1, y1, x2, y2 = bbox
    # x = (x2 + x1) / 2
    # y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    return [x1, y1, width, height]


# Other Helpers
def read_label_map(label_map_path):
    item_id = None
    item_name = None
    items = {}

    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()

            if item_id is not None and item_name is not None:
                items[item_name] = item_id
                item_id = None
                item_name = None

    return items


def COCO_dict():
    coco = {
        "info": {},
        "images": [],
        "annotations": [],
        "license": []
    }
    return coco


# Metrics


# Label Map
def create_label_map(label_map_path):
    label_map = read_label_map(label_map_path)
    label_map['Cormorant DCCO'] = 0
    label_map['Cormorant PECO'] = 0
    label_map['Cormorant DCCO Juvenile'] = 0
    label_map['Cormorant DCCO Adult'] = 0
    label_map['Cormorant PECO Adult'] = 0

    label_map['Nest PECO Occupied'] = 1
    label_map['Nest DCCO Occupied'] = 1
    label_map['Nest DCCO'] = 1
    return label_map

def create_annotation_coco_json(tile_dir, annotation_dir, out_file, label_map):
    all_annotations = [pd.read_csv(f) for f in Path(annotation_dir).glob("*_annos.csv")]
    coco_data = COCO_dict()

    coco_data['annotations'] = list(convert_annotations(all_annotations))
    coco_data['images'] = [{"id": id} for id in
                           set([x['image_id'] for x in coco_data['annotations']])]
    coco_data['categories'] = [{"id": v, "name": k} for k, v in label_map.items()]

    with open(Path(out_file).expanduser(), "w") as f:
        json.dump(coco_data, f)


def create_detection_coco_json(tile_dir, detections_dir, out_file):
    all_detections = [pd.read_csv(f) for f in Path(detections_dir).glob("merged*")]
    results = list(convert_detections(all_detections))
    with open(Path(out_file).expanduser(), "w") as f:
        json.dump(results, f)


def evaluate_from_json_file(annotation_json_file, detection_json_file, out_file=None,
                            max_detections=None, iou_thresholds=None):
    # Ground Truth COCO Object
    cocoGT = COCO(Path(annotation_json_file).expanduser())

    # Detection COCO Object
    with open(Path(detection_json_file).expanduser(), 'r') as d_file:
        detection_results = json.load(d_file)
    cocoDt = cocoGT.loadRes(detection_results)

    # COCO Eval Object
    coco_eval = COCOeval(cocoGt=cocoGT, cocoDt=cocoDt, iouType="bbox")
    if max_detections:
        coco_eval.params.maxDets = max_detections
    if iou_thresholds:
        coco_eval.params.iouThrs = iou_thresholds

    # Evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Write out
    if out_file:
        with open(out_file, "wb") as f:
            pkl.dump(coco_eval, f)

    return coco_eval


def calculate_precisions(coco_eval, iou_threshold=None):
    if iou_threshold:
        iou_index = np.where(np.array(coco_eval.params.iouThrs) == iou_threshold)[0]
        total_map = coco_eval.eval['precision'][iou_index, :, :, 0, 2].mean()
        print(f"mAP{iou_threshold}: \t\t {total_map:.1%}")

    else:
        total_map = coco_eval.eval['precision'][:, :, :, 0, 2].mean()
        print(f"mAP: \t\t\t {total_map:.1%}")

    # Classes
    for i in coco_eval.params.catIds:
        if iou_threshold:
            map_cat = coco_eval.eval['precision'][iou_index, :, i, 0, 2].mean()
            print(f"mAP@{iou_threshold}[{i}]: \t {map_cat:.1%}")
        else:
            map_cat = coco_eval.eval['precision'][:, :, i, 0, 2].mean()
            print(f"mAP[{i}]: \t\t {map_cat:.1%}")


    coco_eval.eval['precision'][0, :, :, 0, 2].mean()  # mAP @ 0.5
    coco_eval.eval['precision'][0, :, 0, 0, 2].mean()  # mAP @ 0.5 Cormorant
    coco_eval.eval['precision'][0, :, 1, 0, 2].mean()  # mAP @ 0.5 Nest


def calculate_recalls(coco_eval, max_detections, iou_threshold=None):
    det_index = np.where(np.array(coco_eval.params.maxDets)==max_detections)[0]

    if iou_threshold:
        iou_index = np.where(np.array(coco_eval.params.iouThrs) == iou_threshold)[0]
        total_ar = coco_eval.eval['recall'][iou_index, :, 0, det_index].mean()
        print(f"AR@{max_detections} @{iou_threshold}IoU: \t {total_ar:.1%}")

    else:
        total_ar = coco_eval.eval['recall'][:, :, 0, det_index].mean()
        print(f"AR@{max_detections}: \t {total_ar:.1%}")

    # Classes
    for i in coco_eval.params.catIds:
        if iou_threshold:
            ar_cat = coco_eval.eval['recall'][iou_index, i, 0, det_index].mean()
            print(f"AR@{max_detections}[{i}] @{iou_threshold}IoU: \t {ar_cat:.1%}")
        else:
            ar_cat = coco_eval.eval['recall'][:, i, 0, det_index].mean()
            print(f"AR@{max_detections}: \t {ar_cat:.1%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('--tiles', help='File path to the directory containing '
                                                   'unannotated tiles. i.e. the directory '
                                                   'containing the containing tifs to be tiled.')
    parser.add_argument('--anno_folder', help='Folder containing manual annotations *_annos.csv.', type=str)
    parser.add_argument('--detections_dir', help='Folder containing merged*.csv detection csv', type=str)
    parser.add_argument('--label_map', help='Path containing label map', type=str)

    parser.add_argument('--out_dir', help='.', type=str)

    args = parser.parse_args()

    # INPUT VARIABLES
    # tile_directory = Path(args.tiles)
    # annotation_dir = Path(args.anno_folder)
    # detections_dir = Path(args.detections_dir)
    # label_map_path = Path(args.label_map)
    WORKSPACE   = Path("/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts")
    WORKSPACE_2 = Path('/Users/jilliana/Documents/rcg_projects/RuthJoy/Cormorants/cormorants-nesting-scripts-bkp')

    tile_directory = WORKSPACE_2 / 'object_detection_scripts/1_preprocessing_annotation_pipeline/tile_tifs/output/2020_SNB/MANUAL_COUNTS/'
    annotation_dir = WORKSPACE / 'object_detection_scripts/4_comparing_manual_counts/manual_counts/output/2020/VALIDATION'
    detections_dir = WORKSPACE / 'object_detection_scripts/3_prediction_pipeline_postprocessing/post_process_detections/output/2020/VALIDATION/snb5_cn_hg_v9'

    label_map_path = WORKSPACE_2 / 'object_detection_scripts/2_training_pipeline/lost_to_tf/input/snb5/label_map.pbtxt'

    # OUTPUT VARIABLES
    # out_dir = Path(args.out_dir)
    out_dir = WORKSPACE / 'object_detection_scripts/4_comparing_manual_counts/calc_mAP/output'
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    annotation_out_file = out_dir / 'annotations.json'
    detection_result_file = out_dir / 'results.json'
        
    # Create label map
    label_map = create_label_map(label_map_path)

    # Annotations
    create_annotation_coco_json(tile_directory,
                                annotation_dir,
                                annotation_out_file,
                                label_map)

    # Detections
    create_detection_coco_json(tile_directory,
                               detections_dir,
                               detection_result_file)

    coco_eval = evaluate_from_json_file(annotation_out_file,
                                        detection_result_file,
                                        max_detections=[20, 200, 400],
                                        iou_thresholds=[0.5],
                                        out_file= out_dir / "coco_eval_iou0.5.csv")

    calculate_precisions(coco_eval, iou_threshold=0.1)
    calculate_recalls(coco_eval, 400)
    calculate_recalls(coco_eval, 400, 0.1)
    calculate_recalls(coco_eval, 200)
    calculate_recalls(coco_eval, 200, 0.1)