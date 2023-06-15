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
#JSON_FILE="$WORKSPACE/snb-2020.json"
# JSON_FILE="$WORKSPACE/snb-2021-no-counts.json"
#JSON_FILE="$WORKSPACE/snb-2021-manual.json"
JSON_FILE="$WORKSPACE/snb-2021-train.json"

jq -c '.[]' $JSON_FILE | while read i; do
    TASK_PATH=$(echo $i | jq -r '.task_path')
    IMAGE=$(echo $i | jq -r '.image')
    IMAGE=$(realpath $IMAGE)

    REPO=cormorants-nesting-scripts
    PIPELINE=$WORKSPACE/$REPO/object_detection_scripts

    # Write Script and Job details to file
    progress $LOGSDIR $LOGSDIR

    # Move to the correct starting point
    echo "LOG STATUS: $TASK_PATH"
    echo "LOG STATUS: Running pipeline"
    cd $PIPELINE

  #  # Tile Image 
  #  echo "LOG STATUS: Tiling Image..."
  #  cd $PIPELINE/tile_tifs
  #  python src/tile_tif.py \
  #      --in_file "$IMAGE" \
  #      --out_dir output/$TASK_PATH/ \
  #      --tile_height $TILE_SIZE \
  #      --tile_width $TILE_SIZE
  #  echo "LOG STATUS: Completed tile_tfs"

  #  # Run Model
  #  echo "LOG STATUS: Predicting..."
  #  cd $PIPELINE/predict
  #  python src/predict.py \
  #      --tiles $PIPELINE/tile_tifs/output/$TASK_PATH \
  #      --exported_model $TRAINED_MODEL \
  #      --out_dir output/$TASK_PATH/ \
  #      --box_thresh 0.1
  #  echo "LOG STATUS: Completed predictions"

    # Post Process Model Results
    echo "LOG STATUS: Post Processing..."
    cd $PIPELINE/post_process_detections
    mkdir -p $PIPELINE/post_process_detections/output/$TASK_PATH/
    python3 src/post_process.py --mask --deduplicate_nests \
      --detections_file $PIPELINE/predict/output/$TASK_PATH/detections.csv \
      --original_pano "$IMAGE" \
      --mask_file $PIPELINE/post_process_detections/input/$TASK_PATH/mask.csv \
      --tile_size 1000 \
      --out_file $PIPELINE/post_process_detections/output/$TASK_PATH/post_processed_detections.csv 
    echo "LOG STATUS: Completed post-processing"

    # Remove Temporary Files
    rm $PIPELINE/tile_tifs/output/$TASK_PATH/*jpg

    # Write Script to Output
    echo "LOG STATUS: Saving output and logs"
    mkdir -p output/$TASK_PATH
    cat $WORKSPACE/run_model_on_new_pano.sh > output/$TASK_PATH/script.bak.log

    # Write module list, environment varibles and script to output
    full_path=$(realpath $0) # $0 is the name of the current script as it was executed

    OUTPUT_FILE=output/$TASK_PATH/script.log
    touch $OUTPUT_FILE

    echo > $OUTPUT_FILE

    module list >> $OUTPUT_FILE

    section_break(){
      (echo && echo && \
      echo "==============================================" && \
      echo && echo ) >> $1
    }
    section_break $OUTPUT_FILE

    # get environment variables that partially contain any of the following keywords
    printenv | grep -E 'NAME|ENVDIR|DEPS|LOGSDIR|SCRATCH|PROJECT|WORKSPACE|TF_MODEL_GARDEN|TF_OBJ_DET|TFHUB_CACHE_DIR' \
      >> $OUTPUT_FILE

    echo >> $OUTPUT_FILE

    printenv | grep -E 'TASK|IMAGE|TILE|REPO|PIPELINE' \
      >> $OUTPUT_FILE

    section_break $OUTPUT_FILE
    echo >> $OUTPUT_FILE

    cat $full_path >> $OUTPUT_FILE
    cp $OUTPUT_FILE $LOGSDIR/ 

done

echo "LOG STATUS: Additional logs may be written to $LOGSDIR"
