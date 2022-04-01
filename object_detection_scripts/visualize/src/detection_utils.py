import ipywidgets
import pandas as pd
import io

import tensorflow as tf
import numpy as np
from object_detection.metrics.coco_evaluation import CocoDetectionEvaluator
from object_detection.core import standard_fields as fields
import pandas as pd
from object_detection.utils import config_util
from object_detection.builders import model_builder
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont
from ast import literal_eval
from pathlib import Path

colour_palette = {0: 'LightGreen',
                  1: 'Plum'}
n = 16
detections = pd.DataFrame()

# Define ipyWidgets
confidence_slider = ipywidgets.FloatSlider(min=0, max=1, step=0.01)
button = ipywidgets.Button(description="Update")
new_photo_button = ipywidgets.Button(description='Show new photos')
output = ipywidgets.Output()
items = [ipywidgets.Image() for i in range(n)]
layout = ipywidgets.Layout(width='100%',
                           grid_template_columns='repeat(4, 200px)',
                           grid_template_rows='repeat(4, 200px)',
                           grid_gap='5px 5px')
image_grid = ipywidgets.GridBox(items, layout=layout)


def draw_detections(detections, threshold, save_dir=None, n=10):
    df_dets = detections.dropna().groupby('image')
    
    drawn_images = []
    
    for image, detection_data in df_dets:
        im = Image.open(image)
        draw = ImageDraw.Draw(im)
        for score, label, box in zip(detection_data['detection_scores'],
                                     detection_data['detection_classes'],
                                     detection_data['detection_boxes']):
            if score >= threshold:
                ymin, xmin, ymax, xmax = literal_eval(box)
                x0 = xmin*im.width
                x1 = xmax*im.width
                y0 = ymin*im.height
                y1 = ymax*im.height
                rect = [x0, y0, x1, y1]

                rect_width = 3
                font_size = 12

                draw.rectangle(rect, outline=colour_palette[label], width=rect_width)
                draw.rectangle([x0, y0, x1, y0-font_size], fill=colour_palette[label])

                font = ImageFont.truetype('Arial Black', size=font_size)
                draw.text(xy=(x0+rect_width, y0),
                          text=f"{label}: {round(score*100)}%",
                          font=font,
                          fill='black',
                          anchor='ls')

        if save_dir:
            img_dir = save_dir.joinpath('img')
            if not img_dir.exists():
                img_dir.mkdir(parents=True)
            img_path = Path(image)
            im.save(img_dir.joinpath(img_path.name))
        else:
            drawn_images.append(im)

    return drawn_images


def on_update_button_clicked(b):
    detections_to_draw = detections[detections['image'].isin(random_photos)]
    
    with output:
        output.clear_output()
        images = draw_detections(detections_to_draw, threshold=confidence_slider.value, n=1)
        for idx, image in enumerate(images):            
            img_byte_arr = io.BytesIO()
            images[idx].save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            items[idx].value = img_byte_arr
        
        
def on_new_photo_button_clicked(b):
    global random_photos
    random_photos = list(detections['image'].drop_duplicates().sample(n))
    
    
def detection_visualizer(dets_file):
    global detections
    detections = pd.read_csv(dets_file)
    random_photos = []
    
    # Associating functions to interactions
    button.on_click(on_update_button_clicked)
    new_photo_button.on_click(on_new_photo_button_clicked)
    
    display(confidence_slider, button, new_photo_button, output, image_grid)

def square(rect):
    """
    Calculates square of rectangle
    """
    return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])

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

def inference_from_bytes(all_images, detection_model, box_th=0.25, nms_th=0.5):
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
    print(f'Ready to start inference on {len(all_images)} images!')
    all_detections = {}
    for image_path, image in tqdm(all_images):
        im = Image.open(BytesIO(image))
        image_np = np.array(im)

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

def tfrecord_2_groundtruth_info(tfrecord_file):
    tfrecord = tf.data.TFRecordDataset(tfrecord_file)
    all_groundtruth_info = []
    raw_images = []
    for raw_record in tfrecord:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        result = {}
        for key, feature in example.features.feature.items():
            kind = feature.WhichOneof('kind')
            result[key] = np.array(getattr(feature, kind).value)
        
        image_id = result['image/filename'][0].decode(encoding='utf-8')
        groundtruth_boxes = np.array(list(zip(result['image/object/bbox/ymin'], 
                                              result['image/object/bbox/xmin'], 
                                              result['image/object/bbox/ymax'], 
                                              result['image/object/bbox/xmax'])))
        groundtruth_dict = {fields.InputDataFields.groundtruth_boxes: groundtruth_boxes, 
                            fields.InputDataFields.groundtruth_classes: result['image/object/class/label']}
        record_info = (image_id, groundtruth_dict)
        all_groundtruth_info.append(record_info)
        raw_images.append((image_id, result['image/encoded']))
        
    return all_groundtruth_info, raw_images

def detections_2_detection_info(detections):
    all_detection_info = []
    for image_id, det_data in detections.items():
        adjusted_classes = [c+1 for c in det_data['detection_classes']]
        detection_info = {fields.DetectionResultFields.detection_boxes: np.array(det_data['detection_boxes']), 
                          fields.DetectionResultFields.detection_scores: np.array(det_data['detection_scores']), 
                          fields.DetectionResultFields.detection_classes: np.array(adjusted_classes)}
        record_info = (image_id, detection_info)

        all_detection_info.append(record_info)
        
    return all_detection_info