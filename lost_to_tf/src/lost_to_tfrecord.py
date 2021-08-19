# **********************************************************************************************************************
# Converting Lost Annotation Format to CSV format in preparation for conversion to TFRecord
# Final csv based on data found here:
#       https://github.com/datitran/raccoon_dataset/blob/master/data/raccoon_labels.csv
# Author: Jillian Anderson (jilliana@sfu.ca)
# **********************************************************************************************************************
import pandas as pd
from PIL import Image, ExifTags
import ast
import numpy as np
import math
import argparse
from pathlib import Path
import tensorflow as tf
from collections import namedtuple, OrderedDict
import io
import dataset_util


def imgpath2filename(img_path):
    splits = img_path.split('/')
    file_name = splits[-1]
    return file_name


def get_img_class(img_name, gbif_species_map):
    gbif_id = int(img_name.split('_')[0])

    species = gbif_species_map[gbif_id]

    return species


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


def train_val_test_split(df, split_ratios):
    """
    Will split the df DataFrame into 3 separate dataframes, thereby creating a training, validation,
    and test set.
    """
    train_ratio, validation_ratio, test_ratio = split_ratios

    # Get the list of image filenames
    filenames = df['filename'].unique()

    # Shuffle the list of image filenames
    np.random.shuffle(filenames)

    # Find the indexes at which to split the filename list into training, validation, and testing
    train_val_split = math.ceil(train_ratio * len(filenames))
    val_test_split = math.ceil((train_ratio + validation_ratio) * len(filenames))

    # Use the split to get the list of image filenames for each split
    train_filenames = filenames[:train_val_split]
    val_filenames = filenames[train_val_split:val_test_split]
    test_filenames = filenames[val_test_split:]
    print(len(train_filenames), len(val_filenames), len(test_filenames))

    # Filter the dataframes based on filenames
    df_train = df[df['filename'].isin(train_filenames)]
    df_val = df[df['filename'].isin(val_filenames)]
    df_test = df[df['filename'].isin(test_filenames)]
    print(len(df_train), len(df_val), len(df_test))

    return df_train, df_val, df_test


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


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


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


def create_tfrecords(annotations, label_map, image_directory, output_path):
    # Take those sets and create a tfrecord
    writer = tf.io.TFRecordWriter(str(output_path))

    grouped = split(annotations, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, label_map, image_directory)
        writer.write(tf_example.SerializeToString())


def lost_to_tfrecord(image_directory, annotation_file, split_ratios, label_map_path, output_path):
    standard_annos = standardize_lost(image_directory, annotation_file)

    training, validation, testing = train_val_test_split(standard_annos, split_ratios)

    label_map = read_label_map(label_map_path)
    create_tfrecords(training, label_map, image_directory,
                     output_path.joinpath('train.tfrecord'))
    create_tfrecords(validation, label_map, image_directory,
                     output_path.joinpath('validation.tfrecord'))
    create_tfrecords(testing, label_map, image_directory,
                     output_path.joinpath('test.tfrecord'))


def create_tf_example(group, label_map, image_directory):
    img_path = str(image_directory.joinpath(group.filename))
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    weights = []  # Important line

    for index, row in group.object.iterrows():
        if int(row['xmin']) > -1:
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(label_map[row['class']])
        else:
            xmins.append(0)
            xmaxs.append(0)
            ymins.append(0)
            ymaxs.append(0)
            classes_text.append('NONE'.encode('utf8'))
            classes.append(0)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        # 'image/object/weight': dataset_util.float_list_feature(weights)  # Important line

    }))
    return tf_example


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes a file path as input and tiles each .tif'
                                                 'file in that directory.')
    parser.add_argument('--img_dir', help='.')
    parser.add_argument('--anno_file', help='.')
    parser.add_argument('--out_path', help='.')
    parser.add_argument('--label_map', help='.')
    parser.add_argument('--splits', nargs='+', type=float, help='.')
    args = parser.parse_args()

    lost_to_tfrecord(image_directory=Path(args.img_dir),
                     annotation_file=Path(args.anno_file),
                     split_ratios=tuple(args.splits),
                     label_map_path=Path(args.label_map),
                     output_path=Path(args.out_path),
                     )

    # e.g.
    # python3 lost_to_tfrecord.py \
    #   --img_dir ../input/DEMO/DEMO_img/ \
    #   --anno_file ../input/DEMO/demo_lost_annotations.csv \
    #   --split 0.7 0.15 0.15 \
    #   --label_map ../input/DEMO/DEMO_label_map.pbtxt \
    #   --out_path ../output/DEMO/
