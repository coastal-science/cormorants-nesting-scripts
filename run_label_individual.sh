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
mkdir -p $LOGSDIR

# User defined variables
TILE_SIZE=1000
ANNO_TILE_SIZE=3000
# JSON_FILE="$WORKSPACE/2023_IWMB_Span2.json"
JSON_FILE="$WORKSPACE/2024_IWMB_Span2.json"
LABEL_STUDIO_FOLDER_PREFIX=2024_nest_pairs

IFS=$'\n'
PREV_IMAGE=""
TASK_NUMBER=0
jq -c '.[]' $JSON_FILE | while read i; do
    
    TASK_PATH=$(echo $i | jq -r '.task_path')
    IMAGE=$(echo $i | jq -r '.image')
    IMAGE=$(realpath $IMAGE)
    
    # Current imagename and with parent directory and detections file 
    filename=$(basename -- $(realpath "$IMAGE"))
    imagename=${filename%.*}
    extension="${filename##*.}"
    extension="png"
    echo $filename $imagename $extension

    # path to table containing individual files detections
    DETECTIONS_CSV=$PIPELINE/4_comparing_manual_counts/draw_final_detections/output/$TASK_PATH/$imagename.csv
    # path to compared pano
    fileparent=$(basename -- $(dirname "$DETECTIONS_CSV"))
    pano_path="$fileparent/$imagename.$extension" 
    
    REPO=cormorants-nesting-scripts
    PIPELINE=$WORKSPACE/$REPO/object_detection_scripts

    # Write Script and Job details to file
    progress $LOGSDIR $LOGSDIR

    # Move to the correct starting point
    echo "LOG STATUS: $TASK_PATH"
    echo "LOG STATUS: Running pipeline"
    echo "LOG STATUS: Image number ${TASK_NUMBER} being processed."
    echo PREV_IMAGE=$PREV_IMAGE
    echo IMAGE=$IMAGE

    if [ $TASK_NUMBER -eq 0 ]; then
        echo "LOG STATUS: Skipping ${TASK_NUMBER} as there is no previous image to compare with."
        TASK_NUMBER=$((TASK_NUMBER + 1))
        PREV_IMAGE=$pano_path
        continue
    fi
    # TASK_NUMBER=$((TASK_NUMBER + 1))
    # PREV_IMAGE=$imagename
    # continue
    cd $PIPELINE

    #
    #
    #

    # Draw Detections 
    echo "LOG STATUS: Producing Label Studio pairs for comparing individual nest detection with panoramas..."
    cd $PIPELINE/4_comparing_manual_counts/compare_detections_labelstudio/
    mkdir -p output/$TASK_PATH
    PARENT_PATH=$(dirname $PIPELINE/3_prediction_pipeline_postprocessing/post_process_detections/input/$TASK_PATH/) # model directory (e.g. snb5_cn_hg_v9)
    # PARENT_PATH=$(dirname $PARENT_PATH) # location/site directory (e.g. IWMB)
    
    time python src/label_studio.py \
        --input_file $PIPELINE/4_comparing_manual_counts/draw_final_detections/output/$TASK_PATH/$imagename.csv \
        --right $PREV_IMAGE \
        --folder_prefix $LABEL_STUDIO_FOLDER_PREFIX \
        --out_folder output/$TASK_PATH \
        || fail "Label Studio Parsing failed" $?
    
    echo "LOG STATUS: Completed draw_detections"

    echo "===> LOG STATUS: Completed $TASK_PATH"

    TASK_NUMBER=$((TASK_NUMBER + 1))
    PREV_IMAGE=$pano_path
    
done

echo "LOG STATUS: Completed all tasks."
echo "LOG STATUS: Additional logs may be written to $LOGSDIR"

deactivate
