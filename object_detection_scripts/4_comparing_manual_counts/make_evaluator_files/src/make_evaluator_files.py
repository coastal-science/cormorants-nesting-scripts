from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import ast
from pathlib import Path

Image.MAX_IMAGE_PIXELS = 3000000000

colour_palette = {0: (255, 0, 0, 150),  # Red for Cormorants,
                  1: (255, 255, 0, 150),   # Plum for Nests
                  }

rect_line_width = 4
font_size = 7


def calc_tile_dimension(tile_num, original_img_size, ideal_tile_size):
    if original_img_size - (tile_num * ideal_tile_size) < ideal_tile_size:
        tile_size = original_img_size - (tile_num * ideal_tile_size)
    else:
        tile_size = ideal_tile_size

    return tile_size


def locate_detection_in_tile(detection, tile_num, original_img_size, reduced_img_size, ideal_tile_size):
    reduce_factor = original_img_size / reduced_img_size
    tile_size = calc_tile_dimension(tile_num, original_img_size, ideal_tile_size)
    location_in_tile = (tile_size * detection) / reduce_factor
    return location_in_tile


def locate_tile_in_image(tile_num, ideal_tile_size, reduced_image_size, original_image_size):
    reduce_factor = original_image_size / reduced_image_size
    return tile_num * ideal_tile_size / reduce_factor


def draw_detection(draw, detection, ideal_tile_size, original_image_size):
    image, score, label, box, id_num = detection
    box = ast.literal_eval(box)

    tile_y, tile_x,  _ = Path(image).name.split('.')
    tile_y = int(tile_y)
    tile_x = int(tile_x)
    ideal_tile_width, ideal_tile_height = ideal_tile_size
    small_width, small_height = draw.im.size

    original_width, original_height = original_image_size

    ymin, xmin, ymax, xmax = box
    x0 = locate_detection_in_tile(xmin, tile_x, original_width, small_width, ideal_tile_width) + \
         locate_tile_in_image(tile_x, ideal_tile_width, small_width, original_width)
    x1 = locate_detection_in_tile(xmax, tile_x, original_width, small_width, ideal_tile_width) + \
         locate_tile_in_image(tile_x, ideal_tile_width, small_width, original_width)
    y0 = locate_detection_in_tile(ymin, tile_y, original_height, small_height, ideal_tile_height) + \
         locate_tile_in_image(tile_y, ideal_tile_height, small_height, original_height)
    y1 = locate_detection_in_tile(ymax, tile_y, original_height, small_height, ideal_tile_height) + \
         locate_tile_in_image(tile_y, ideal_tile_height, small_height, original_height)
    rect = [x0, y0, x1, y1]


    draw.rectangle(rect, outline=colour_palette[label], width=rect_line_width)
    draw.rectangle([x0, y0, x1, y0 - font_size], fill=colour_palette[label])

    font = ImageFont.truetype('Arial Black', size=font_size)
    draw.text(xy=(x0 + rect_line_width, y0),
              text=f"{id_num}({label})",
              font=font,
              fill='black',
              anchor='ls')

    return x0, y0


def create_final_dataframe(df):
    df['approximate_location_x_y'] = df['box_location'].apply(lambda x: (round(x[0]), round(x[1])))
    df = df[['detection_scores', 'detection_classes', 'approximate_location_x_y']]
    df['TRUE_POSITIVE'] = None
    return df


def foo(img_file, detections_file, rescale_factor, out_dir):
    fullsize_im = Image.open(img_file)
    im = fullsize_im.reduce(factor=rescale_factor)
    draw = ImageDraw.Draw(im, "RGBA")

    df = pd.read_csv(detections_file)
    dets = df[~df['detection_scores'].isna()].reset_index()

    det_locations = []
    for detection in zip(dets['image'], dets['detection_scores'],
                         dets['detection_classes'].astype(int), dets['detection_boxes'],
                         dets.index):
        det_locations.append(draw_detection(draw, detection, (1000, 1000), fullsize_im.size))
    dets['box_location'] = det_locations
    df = create_final_dataframe(dets)
    df.to_csv(out_dir + '/full_image_detections.csv')
    im.save(out_dir + '/full_image_detections.jpg')


if __name__ == '__main__':
    img_file = 'cormorants-nesting-scripts/object_detection_scripts/make_evaluator_files/input/SNB2020/SNB_24062020/SNB_24062020_70756x35604.tif'
    detections_file = 'cormorants-nesting-scripts/object_detection_scripts/make_evaluator_files/input/SNB2020/SNB_24062020/snb3_centernet101_512_v1-0.1/detections.csv'
    rescale_factor = 5
    out_dir = 'cormorants-nesting-scripts/object_detection_scripts/make_evaluator_files/output/test2'
    foo(img_file, detections_file, rescale_factor, out_dir)

    img_file = 'cormorants-nesting-scripts/object_detection_scripts/make_evaluator_files/input/SNB2020/SNB_08062020/SNB_08062020_74576x33620.tif'
    detections_file = 'cormorants-nesting-scripts/object_detection_scripts/make_evaluator_files/input/SNB2020/SNB_08062020/snb3_centernet101_512_v1-0.1/detections.csv'
    rescale_factor = 5
    out_dir = 'cormorants-nesting-scripts/object_detection_scripts/make_evaluator_files/output/SNB_06082020'
    foo(img_file, detections_file, rescale_factor, out_dir)