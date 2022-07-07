#!/bin/bash

YEAR_DIR="../input/2021_SNB/TEST/*"
OUT_DIR="../output/SNB_2021/TEST/"
for d in $YEAR_DIR ; do
    DNAME=$(basename -- $d)
    python3 post_process.py \
      --detections_file $YEAR_DIR$DNAME/snb3v1/detections.csv \
      --mask_file $YEAR_DIR$DNAME/span2_bridge_mask.csv \
      --out_file $OUT_DIR${DNAME}_snb3_v1_detections_pp3.csv
done
