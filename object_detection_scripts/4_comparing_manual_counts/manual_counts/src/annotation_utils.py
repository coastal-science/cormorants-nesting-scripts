import copy
import re
import ast
from pathlib import Path
from PIL import Image
import pandas as pd
from shapely.geometry import box


class BoundingBox:
    def __init__(self, corners=None, lost=None, coco=None):
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0

        if corners:
            self.load_four_corners(*corners)
        elif lost:
            self.load_lost_format(**lost)
        elif coco:
            self.load_coco_format(*coco)

    def load_four_corners(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

        return self

    def load_lost_format(self, x, y, w, h):
        """Note, in LOST format, x & y refer to x_center & y_center"""
        self.x0 = x - w / 2
        self.y0 = y - h / 2
        self.x1 = x + w / 2
        self.y1 = y + h / 2

        return self

    def load_coco_format(self, x0, y0, w, h):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x0 + w
        self.y1 = y0 + h

    def to_coco(self):
        return self.x0, self.y0, self.x1-self.x0, self.y1-self.y0

    def four_corners(self):
        return self.x0, self.y0, self.x1, self.y1

    def plt_dims(self):
        width = self.x1 - self.x0
        height = self.y1 - self.y0

        return (self.x0, self.y0), width, height

    def lost_dims(self):
        x_center = (self.x0 + self.x1) / 2
        y_center = (self.y0 + self.y1) / 2

        width = self.x1 - self.x0
        height = self.y1 - self.y0

        return x_center, y_center, width, height

    def geom(self):
        return box(self.x0, self.y0, self.x1, self.y1)

    def area(self):
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def resize(self, tile_name, tile_size, rescale_factor):
        result = copy.deepcopy(self)
        tile_w, tile_h = tile_size
        tile_y, tile_x, _ = tile_name.split('.')
        tile_y = int(tile_y)
        tile_x = int(tile_x)

        result.x0 = (tile_x + result.x0) * tile_w / rescale_factor
        result.y0 = (tile_y + result.y0) * tile_h / rescale_factor
        result.x1 = (tile_x + result.x1) * tile_w / rescale_factor
        result.y1 = (tile_y + result.y1) * tile_h / rescale_factor

        return result

    def merge(self, other):
        self.x0 = min(self.x0, other.x0)
        self.y0 = min(self.y0, other.y0)
        self.x1 = max(self.x1, other.x1)
        self.y1 = max(self.y1, other.y1)

        return self



def x1_y1_x2_y2_conversion(bbox):
    x1, y1, x2, y2 = bbox
    # x = (x2 + x1) / 2
    # y = (y1 + y2) / 2
    x_origin = x1
    y_origin = y1
    width = x2 - x1
    height = y2 - y1

    return [x_origin, y_origin, width, height]


def resize_annotation(row, tile_directory, rescale_factor):
    raw_anno = ast.literal_eval(row['detection_boxes'])
    raw_path = row['img.img_path']
    img_date = re.search("[0-9]{8}", raw_path).group()
    tile_name = Path(raw_path).name

    tile = Image.open(tile_directory.joinpath(img_date).joinpath(tile_name))
    tile_y, tile_x, _ = tile_name.split('.')
    tile_w, tile_h = tile.size

    tile_x0, tile_y0, tile_x1, tile_y1 = raw_anno
    x0 = tile_x0 * tile_w / rescale_factor
    y0 = tile_y0 * tile_h / rescale_factor
    x1 = tile_x1 * tile_w / rescale_factor
    y1 = tile_y1 * tile_h / rescale_factor

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
    all_annotations = pd.concat(dataframes).dropna(subset=['detection_boxes']).apply(
        annotation_to_json, axis=1)

    return all_annotations


def annotation_to_json(row, tile_directory, label_map):
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

class Annotation:
    def __init__(self, lost_row=None, coco_det_row=None, label_map=None, tile_directory=None):
        self.id = None
        self.image_id = None
        self.category_id = None
        self.iscrowd = None
        self.bbox = None
        self.area = None
        self.source = None
        self.detection_id = None

        if lost_row is not None:
            self.load_lost(lost_row, label_map, tile_directory)
        if coco_det_row is not None:
            self.load_coco_detection(coco_det_row)

    def load_lost(self, lost_row, label_map, tile_directory):
        self.id = lost_row['anno.idx']
        self.image_id = int(re.search("[0-9]{8}", lost_row['img.img_path']).group())
        self.category_id = label_map(ast.literal_eval(lost_row['anno.lbl.name'])[0])
        self.iscrowd = 0

        # Load Lost Row into a Bounding Box
        tile_name = Path(lost_row['img.img_path']).name
        tile = Image.open(tile_directory.joinpath(str(self.image_id)).joinpath(tile_name))
        local_bbox = BoundingBox(lost=ast.literal_eval(lost_row['anno.data']))
        self.bbox = local_bbox.resize(tile_name, tile.size, rescale_factor=1)
        self.area = self.bbox.area()
        self.source = 'ground truth annotation'

    def load_coco_detection(self, coco_row):
        self.id = -1 * coco_row['id']
        self.detection_id = coco_row['id']
        self.image_id = coco_row['image_id']
        self.category_id = coco_row['category_id']
        self.iscrowd = 0

        # Load into a Bounding Box
        self.bbox = BoundingBox(coco=ast.literal_eval(coco_row['bbox']))
        self.area = coco_row['area']
        self.source = 'false positive detection'

    def coco_anno_dict(self):
        annotation_dict = {
            'id': self.id,
            'image_id': self.image_id,
            'category_id': self.category_id,
            'area': self.area,
            'bbox': self.bbox.to_coco(),
            'iscrowd': self.iscrowd
        }

        return annotation_dict

    def merge(self, other):
        # Make sure all categories are the same
        if self.category_id == other.category_id:
            self.bbox = self.bbox.merge(other.bbox)
        else:
            print(f"\t! Could not merge {self.id:.0f} & {other.id:.0f} from {self.image_id} as they have different labels !")
