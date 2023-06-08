#!/bin/bash
# Example bash script that can be used to create a pair of RMSE & count comparison charts.

# **********************
# User-Defined Variables
# **********************
# Input Variables
IN_DIR="../input/SNB_2021/TEST-COUNTS"
DETECTIONS="snb5_cn_hg_v9_detections"
FILEMAP="snb5_cn_hg_v9_file_map.json"
# Processing Variables
THRESH_CORM=0.2 # From SNB5 CN HG v9 on 2020
THRESH_NEST=0.2 # FROM SNB5 CN HG v9 on 2020
# Output Variables
OUT_DIR="../output/SNB_2021/TEST-COUNTS/snb5_cn_hg_TEST"
RMSE_FILE='rmse.png'
COUNTS_FILE='counts.png'

echo "Creating RMSE Plot & CSV..."
python3 compare_counts.py \
--true_counts $IN_DIR/manual_counts_verified.csv \
--detections_dir $IN_DIR/$DETECTIONS \
--file_map $IN_DIR/$FILEMAP \
--out_path $OUT_DIR/$RMSE_FILE \
--plot_type rmse \
--save_raw_csv

echo "Creating Count Plot & CSV..."
python3 compare_counts.py \
  --true_counts $IN_DIR/manual_counts_verified.csv \
  --detections_dir $IN_DIR/$DETECTIONS \
  --file_map $IN_DIR/$FILEMAP \
  --out_path $OUT_DIR/$COUNTS_FILE \
  --plot_type counts \
  --threshold "{0.0: $THRESH_CORM, 1.0: $THRESH_NEST}"

echo "Completed"
