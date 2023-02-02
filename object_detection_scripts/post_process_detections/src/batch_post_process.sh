#!/bin/bash

YEAR_DIR="../input/2020_SNB/VALIDATION/*"
OUT_DIR="../output/SNB_2020/VALIDATION/PP2/"
for d in $YEAR_DIR ; do
    DNAME=$(basename -- $d)
    python3 post_process.py \
      --detections_file $d/snb6_cn_v1/detections.csv \
      --mask_file $d/span2_mask.csv \
      --original_pano $d/original-pano.tif \
      --tile_size 1000 \
      --out_file $OUT_DIR${DNAME}_snb6_cn_v1_detections_pp2.csv
done
