"""
e.g.
resize_image.py \
  --img_file ../input/2021_SNB/SNB_14_10062021.tif \
  --out_file ../output/2021_SNB/SNB_14_10062021_REDUCED_10.jpg \
  --rescale_factor 10
"""
import argparse
from PIL import Image

Image.MAX_IMAGE_PIXELS = 3000000000


def main(img_file, out_file, rescale_factor=10):
    fullsize_im = Image.open(img_file)
    im = fullsize_im.reduce(factor=rescale_factor)
    rgb_im = im.convert('RGB')
    rgb_im.save(out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_file', help='.', type=str)
    parser.add_argument('--out_file', help='.', type=str)
    parser.add_argument('--rescale_factor', help='.', type=int)
    args = parser.parse_args()

    main(args.img_file, args.out_file, args.rescale_factor)
