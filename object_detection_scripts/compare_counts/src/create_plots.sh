#!/bin/bash

IN_DIR="../input/SNB_2020/VALIDATION/"
DETECTIONS="snb6_cn_v1_detections"
FILEMAP="snb6_cn_v1_file_map.json"
OUT_DIR="../output/SNB_2020/VALIDATION/snb6_cn_v1"
RMSE_FILE='rmse.png'
COUNTS_FILE='counts.png'
THRESH_CORM=0.35
THRESH_NEST=0.20

echo "Creating RMSE Plot & CSV..."
python3 compare_counts.py \
--true_counts $IN_DIR/manual_counts_verified.csv \
--detections_dir $IN_DIR/$DETECTIONS \
--file_map $IN_DIR/$FILEMAP \
--out_path $OUT_DIR/$RMSE_FILE \
--plot_type rmse

echo "Creating Count Plot & CSV..."
python3 compare_counts.py \
  --true_counts $IN_DIR/manual_counts_verified.csv \
  --detections_dir $IN_DIR/$DETECTIONS \
  --file_map $IN_DIR/$FILEMAP \
  --out_path $OUT_DIR/$COUNTS_FILE \
  --plot_type counts \
  --threshold "{0.0: $THRESH_CORM, 1.0: $THRESH_NEST}"

echo "Completed"
