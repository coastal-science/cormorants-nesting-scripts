"""
Based off of ...

e.g.
python3 model_informed_data_selector.py \
  --unannotated_pool ../input/snb1/unannotated_pool \
  --exported_model ../input/snb1/centernet_resnet101_512_v3 \
  --tf_label_map ../input/snb1/label_map.pbtxt \
  --lost_label_tree ../input/snb1/cormorants_labels.csv \
  --out_dir ../output/snb1/centernet_resnet101_512_v3_TEST
  --box_thresh 0.1
  --filter_max_score 0.2

"""
import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import operator

# importing all scripts that will be needed to export your model and use it for inference
from object_detection.utils import config_util
from object_detection.builders import model_builder

def load_model(exported_model):
    configs = config_util.get_configs_from_pipeline_file(str(exported_model.joinpath('pipeline.config')))
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config,
                                          is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(str(exported_model.joinpath('checkpoint/ckpt-0'))).expect_partial()
    return detection_model, ckpt


def detect_fn(image, detection_model):
    """
    Detect objects in image.

    Args:
      image: (tf.tensor): 4D input image

    Returs:
      detections (dict): predictions that model made
    """

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


def nms(rects, thd=0.5):
    """
    Filter rectangles
    rects is array of oblects ([x1,y1,x2,y2], confidence, class)
    thd - intersection threshold (intersection divides min square of rectange)
    """
    out = []

    remove = [False] * len(rects)

    for i in range(0, len(rects) - 1):
        if remove[i]:
            continue
        inter = [0.0] * len(rects)
        for j in range(i, len(rects)):
            if remove[j]:
                continue
            inter[j] = intersection(rects[i][0], rects[j][0]) / min(square(rects[i][0]), square(rects[j][0]))

        max_prob = 0.0
        max_idx = 0
        for k in range(i, len(rects)):
            if inter[k] >= thd:
                if rects[k][1] > max_prob:
                    max_prob = rects[k][1]
                    max_idx = k

        for k in range(i, len(rects)):
            if (inter[k] >= thd) & (k != max_idx):
                remove[k] = True

    for k in range(0, len(rects)):
        if not remove[k]:
            out.append(rects[k])

    boxes = [box[0] for box in out]
    scores = [score[1] for score in out]
    classes = [cls[2] for cls in out]
    return boxes, scores, classes


def intersection(rect1, rect2):
    """
    Calculates square of intersection of two rectangles
    rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
    return: square of intersection
    """
    x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
    overlapArea = x_overlap * y_overlap;
    return overlapArea


def inference_as_raw_output(path2images,
                            detection_model,
                            box_th=0.25,
                            nms_th=0.5,
                            # to_file=False,
                            # data=None,
                            # path2dir=False
                            ):
    """
    Function that performs inference and return filtered predictions

    Args:
      all_images: an array with pathes to images
      box_th: (float) value that defines threshold for model prediction. Consider 0.25 as a value.
      nms_th: (float) value that defines threshold for non-maximum suppression. Consider 0.5 as a value.
      path2image + (x1abs, y1abs, x2abs, y2abs, score, conf) for box in boxes

    Returns:
      detections (dict): filtered predictions that model made
    """
    all_images = list(path2images.glob("*"))
    print(f'Ready to start inference on {len(all_images)} images!')

    all_detections = {}
    for image_path in tqdm(all_images):
        image_np = load_image_into_numpy_array(image_path)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor, detection_model)

        # checking how many detections we got
        num_detections = int(detections.pop('num_detections'))

        # filtering out detection in order to get only the one that are indeed detections
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # defining what we need from the resulting detection dict that we got from model output
        key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']

        # filtering out detection dict in order to get only boxes, classes and scores
        detections = {key: value for key, value in detections.items() if key in key_of_interest}

        if box_th:  # filtering detection if a confidence threshold for boxes was given as a parameter
            for key in key_of_interest:
                scores = detections['detection_scores']
                current_array = detections[key]
                filtered_current_array = current_array[scores > box_th]
                detections[key] = filtered_current_array

        if nms_th:  # filtering rectangles if nms threshold was passed in as a parameter
            # creating a zip object that will contain model output info as
            output_info = list(zip(detections['detection_boxes'],
                                   detections['detection_scores'],
                                   detections['detection_classes']
                                   )
                               )
            boxes, scores, classes = nms(output_info)

            detections['detection_boxes'] = boxes  # format: [y1, x1, y2, x2]
            detections['detection_scores'] = scores
            detections['detection_classes'] = classes

        all_detections[image_path] = detections

    return all_detections


def square(rect):
    """
    Calculates square of rectangle
    """
    return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])


def tile_id(tile_path):
    return f"{tile_path.parent.name}.{tile_path.name}"


def find_images_with_detections(detections):
    images_with_detections = [path for path, det in detections.items() if
                              len(det['detection_scores']) > 0]

    return images_with_detections


def read_tf_label_map(label_map_path):
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
                items[item_id] = item_name
                item_id = None
                item_name = None

    return items


def read_lost_label_tree(label_tree_path):
    df = pd.read_csv(label_tree_path)
    return {n: i for i, n in zip(df['idx'], df['name'])}


def create_tf2lost_map(tf_label_map, lost_label_tree):
    # Create a dictionary that maps a TF ID to a label name
    tf2label = read_tf_label_map(tf_label_map)

    # Create a dictionary that maps a label name to a LOST ID
    label2lost = read_lost_label_tree(lost_label_tree)

    # Create a mapping from TF Ids to LOST Ids. 1 is subtracted from TF ids to match the labels in
    # the detection output (start at 0).
    tf2lost = {tf_id-1: label2lost[tf_name] for tf_id, tf_name in tf2label.items()}

    return tf2lost


def calc_num_detections(detections):
    num_detections_per_image = [len(d['detection_scores']) for d in detections.values()]
    total_objects_detected = sum(num_detections_per_image)
    return total_objects_detected


def filter_detections(detections, score_filter):
    (filter_func, filter_op), filter_thresh = score_filter
    non_empty = {f: d for f, d in detections.items() if len(d['detection_scores']) > 0}
    filtered_detections = {}
    for file, data in non_empty.items():
        filtering_val = filter_func(data['detection_scores'])
        if filter_op(filtering_val, filter_thresh):
            filtered_detections[file]= data

    return filtered_detections


def select_tiles(exported_model_dir, unannotated_pool_dir, out_dir, tf_label_map, lost_label_tree, box_thresh, score_filter=None):
    # Load the Model
    detection_model, checkpoint = load_model(exported_model_dir)

    # Use the detection model to conduct inference on the unannotated pool
    detections = inference_as_raw_output(unannotated_pool_dir,
                                         detection_model,
                                         box_th=box_thresh)

    # Filter the detections based on the provided score filter
    if score_filter:
        detections = filter_detections(detections, score_filter)

    # Select images from the unannotated pool to annotate.
    images = find_images_with_detections(detections)

    # Create the TF Class ID to Lost ID Map
    tf2lost_map = create_tf2lost_map(tf_label_map, lost_label_tree)

    # Write detections to a file
    save_images(images, out_dir)
    save_detections(detections, out_dir, tf2lost_map)

    print(f"Detected {calc_num_detections(detections)} objects in {len(set(images))} images")


def save_images(tiles, out_dir):
    # Create the output tif_directory
    img_out_dir = out_dir.joinpath('img/')
    if not img_out_dir.exists():
        img_out_dir.mkdir(parents=True)

    # Symlink each image
    for tile in tiles:
        tile_out_path = img_out_dir.joinpath(tile_id(tile.resolve()))
        tile_out_path.symlink_to(tile.resolve())


def get_lost_bboxes(bbox):
    ymin, xmin, ymax, xmax = bbox

    lost_bbox = {'x': (xmin + xmax) / 2,
                 'y': (ymin + ymax) / 2,
                 'w': xmax - xmin,
                 'h': ymax - ymin}

    return lost_bbox


def split_out_detections(detections):
    individual_detections = []
    for path in detections:
        for idx in range(len(detections[path]['detection_classes'])):
            individual_detections.append({'path': path,
                                          'detection_scores': detections[path]['detection_scores'][idx],
                                          'detection_classes': detections[path]['detection_classes'][idx],
                                          'detection_boxes': detections[path]['detection_boxes'][idx]
                                          })
    return individual_detections


def save_detections(detections, out_dir, tf2lost_label_map):
    # Create the output tif_directory
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    # Find the detections output path
    detection_out_path = out_dir.joinpath('detections.csv')

    #
    detections = {path: det for path, det in detections.items() if len(det['detection_scores']) > 0}
    indiv_detections = split_out_detections(detections)
    detection_df = pd.DataFrame()
    detection_df['img.img_path'] = [tile_id(d['path'].resolve()) for d in indiv_detections]
    detection_df['anno.dtype'] = ['bbox' for _ in indiv_detections]
    detection_df['anno.data'] = [get_lost_bboxes(d['detection_boxes']) for d in indiv_detections]
    detection_df['anno.lbl.idx'] = [tf2lost_label_map[d['detection_classes']] for d in indiv_detections]

    # detection_df['detection_scores'] = [d['detection_scores'] for d in indiv_detections]

    detection_df.to_csv(detection_out_path)


if __name__ == '__main__':
    filters = {'max': (max, operator.le)}

    parser = argparse.ArgumentParser(description='Takes a file path as input and tiles each .tif'
                                                 'file in that tif_directory.')
    parser.add_argument('--unannotated_pool', help='File path to the tif_directory containing '
                                                   'unannotated tiles. i.e. the tif_directory '
                                                   'containing the  containing tifs to be tiled.')
    parser.add_argument('--exported_model', help='File path to the tif_directory containing the saved '
                                                 'model')
    parser.add_argument('--tf_label_map', help='File path to the file')
    parser.add_argument('--lost_label_tree', help='.')
    parser.add_argument('--out_dir', help='.')
    parser.add_argument('--box_thresh', help='.', type=float)
    parser.add_argument('--filter', help='.', type=str)
    parser.add_argument('--filter_thresh', help='.', type=float)

    args = parser.parse_args()

    if args.filter:
        score_filter = (filters[args.filter], args.filter_thresh)
    else:
        score_filter = None

    select_tiles(exported_model_dir=Path(args.exported_model),
                 unannotated_pool_dir=Path(args.unannotated_pool),
                 out_dir=Path(args.out_dir),
                 tf_label_map=Path(args.tf_label_map),
                 lost_label_tree=Path(args.lost_label_tree),
                 box_thresh=args.box_thresh,
                 score_filter=score_filter)
