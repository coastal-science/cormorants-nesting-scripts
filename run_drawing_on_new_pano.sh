#!/bin/bash
#SBATCH --cpus-per-task=1  		# Look at Cluster docs for CPU/GPU ratio 
#SBATCH --mem=16000M       		# Memory proportional to GPUs: 32000 Cedar
#SBATCH --time=0-1:00:00     	# DD-HH:MM:SS

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

JSON_FILE="$WORKSPACE/snb-2021-train.json"
JSON_FILE="$WORKSPACE/astoria-2023.json"

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

    # Draw Detections 
    echo "LOG STATUS: Draw Detections..."
    cd $PIPELINE/4_comparing_manual_counts/draw_final_detections
    python src/draw_detections.py \
        --img_file "$IMAGE" \
        --detections_file $PIPELINE/3_prediction_pipeline_postprocessing/post_process_detections/output/$TASK_PATH/post_processed_detections_masked.csv \
        --mask_file "" \
        --tile_size $TILE_SIZE \
        --threshold_dict '{"0.0": 0.2, "1.0": 0.2}' \
        --rescale_factor 4 \
        --out_file output/$TASK_PATH \
        || fail "Draw Detections failed" $?
        # --ground_truth_file \
        # --tile_directory \
    echo "LOG STATUS: Completed draw_detections"
    
   


    echo "===> LOG STATUS: Completed $TASK_PATH"


done

echo "LOG STATUS: Completed all tasks."
echo "LOG STATUS: Additional logs may be written to $LOGSDIR"
