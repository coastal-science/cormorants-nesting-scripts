#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2  		# Look at Cluster docs for CPU/GPU ratio 
#SBATCH --mem=16000M       		# Memory proportional to GPUs: 32000 Cedar
#SBATCH --time=0-1:00:00     		# DD-HH:MM:SS
#SBATCH --mail-user=jilliana@sfu.ca
#SBATCH --mail-type=ALL

# Load functions
. utils.sh # progress()

# Prepare Environment
module load python/3.7 gcc/9.3.0 arrow/2.0.0 cuda cudnn geos
source config-env.sh

source ${ENVDIR}/bin/activate && \
echo && echo "LOG STATUS: Activated environment ""$NAME"

export LOGSDIR="${WORKSPACE}/logs_${SLURM_JOB_ID}" && \
mkdir -p $LOGSDIR

# User defined variables
TILE_SIZE=1000
TRAINED_MODEL=$WORKSPACE/exported_models/snb5/centernet_hg104_512/v9/
# JSON_FILE="$WORKSPACE/snb-2020.json"
# JSON_FILE="$WORKSPACE/snb-2021-manual.json"
# JSON_FILE="$WORKSPACE/snb-2021-no-counts.json"
JSON_FILE="$WORKSPACE/snb-2021-train.json"
JSON_FILE="$WORKSPACE/astoria-2023.json"

jq -c '.[]' $JSON_FILE | while read i; do
    TASK_PATH=$(echo $i | jq -r '.task_path')
    IMAGE=$(echo $i | jq -r '.image')
    IMAGE=$(realpath $IMAGE)

    REPO=cormorants-nesting-scripts
    PIPELINE=$WORKSPACE/$REPO/object_detection_scripts

    # Write Script and Job details to file
    progress . $LOGSDIR

    # Move to the correct starting point
    echo "LOG STATUS: $TASK_PATH"
    echo "LOG STATUS: Running pipeline"
    cd $PIPELINE

   # Tile Image 
   echo "LOG STATUS: Tiling Image..."
   cd $PIPELINE/1_preprocessing_annotation_pipeline/tile_tifs
   python src/tile_tifs.py \
      --tif_file "$IMAGE" \
      --out_dir output/$TASK_PATH/ \
      --tile_height $TILE_SIZE \
      --tile_width $TILE_SIZE \
      || fail "Tiling failed" $?
   echo "LOG STATUS: Completed tile_tfs"
  
   # Run Model
   echo "LOG STATUS: Predicting..."
   cd $PIPELINE/3_prediction_pipeline_postprocessing/predict
   python src/predict.py \
       --tiles $PIPELINE/1_preprocessing_annotation_pipeline/tile_tifs/output/$TASK_PATH/ \
       --exported_model $TRAINED_MODEL \
       --out_dir output/$TASK_PATH/ \
       --box_thresh 0.1 || fail "Predicting failed" $?
   echo "LOG STATUS: Completed predictions"

    # Post Process Model Results
    echo "LOG STATUS: Post Processing..."
    cd $PIPELINE/3_prediction_pipeline_postprocessing/post_process_detections
    mkdir -p $PIPELINE/3_prediction_pipeline_postprocessing/post_process_detections/output/$TASK_PATH/
    python3 src/post_process.py \
      --mask \
      --detections_file $PIPELINE/3_prediction_pipeline_postprocessing/predict/output/$TASK_PATH/detections.csv \
      --original_pano "$IMAGE" \
      --mask_file $PIPELINE/3_prediction_pipeline_postprocessing/post_process_detections/input/$TASK_PATH/mask.csv \
      --tile_size 1000 \
      --out_file $PIPELINE/3_prediction_pipeline_postprocessing/post_process_detections/output/$TASK_PATH/post_processed_detections_masked.csv \
      --deduplicate_nests \
      --merge_duplicate_nests \
      || fail "Post Processing failed" $?

    echo "LOG STATUS: Completed post-processing"

    # Remove Temporary Files
    # rm $PIPELINE/1_preprocessing_annotation_pipeline/tile_tifs/output/$TASK_PATH/*jpg

    echo "===> LOG STATUS: Completed $TASK_PATH"


done

echo "LOG STATUS: Completed all tasks."
echo "LOG STATUS: Additional logs may be written to $LOGSDIR"
