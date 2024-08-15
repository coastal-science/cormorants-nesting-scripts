#!/bin/bash
#SBATCH --cpus-per-task=2  		# Look at Cluster docs for CPU/GPU ratio 
#SBATCH --mem=0       		# Memory proportional to GPUs: 32000 Cedar
#SBATCH --time=05:00:00     	# DD-HH:MM:SS

# Load functions
. utils.sh # progress()

# # Prepare Environment
module load StdEnv/2020 python/3.7 gcc/9.3.0 arrow/2.0.0 proj/8.0.0 cuda cudnn geos
source config-env.sh

source ${ENVDIR}/bin/activate && \
echo && echo "LOG STATUS: Activated environment ""$NAME"

export LOGSDIR="${WORKSPACE}/logs_${SLURM_JOB_ID}" && \
# mkdir -p $LOGSDIR

# User defined variables
TILE_SIZE=1000
ANNO_TILE_SIZE=3000
# JSON_FILE="$WORKSPACE/2023_IWMB_Span2.json"
JSON_FILE="$WORKSPACE/2024_IWMB_Span2.json"

IFS=$'\n'
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

    #
    #
    #

    # Draw Detections 
    echo "LOG STATUS: Draw Detections..."
    cd $PIPELINE/4_comparing_manual_counts/draw_final_detections
    mkdir -p output/$TASK_PATH
    PARENT_PATH=$(dirname $PIPELINE/3_prediction_pipeline_postprocessing/post_process_detections/input/$TASK_PATH/) # model directory (e.g. snb5_cn_hg_v9)
    # PARENT_PATH=$(dirname $PARENT_PATH) # location/site directory (e.g. IWMB)
    time python src/draw_detections.py \
        --img_file "$IMAGE" \
        --detections_file $PIPELINE/3_prediction_pipeline_postprocessing/post_process_detections/output/$TASK_PATH/post_processed_detections_masked.csv \
        --mask_file $PARENT_PATH/mask.csv \
        --tile_size $TILE_SIZE \
        --threshold_dict '{"0.0": 0.2, "1.0": 0.2}' \
        --rescale_factor 6 \
        --anno_tile_size $ANNO_TILE_SIZE \
        --individual_class 1 \
        --out_file output/$TASK_PATH \
        || fail "Draw Detections failed" $?
        # --ground_truth_file \
        # --tile_directory \
        # --no-indv \
        # --no-full \

    echo "LOG STATUS: Completed draw_detections"

    echo "===> LOG STATUS: Completed $TASK_PATH"
    
done

echo "LOG STATUS: Completed all tasks."
echo "LOG STATUS: Additional logs may be written to $LOGSDIR"

deactivate
