#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1  		# Look at Cluster docs for CPU/GPU ratio 
#SBATCH --mem=16G       		# Memory proportional to GPUs: 32000 Cedar
#SBATCH --time=0-0:30:00     		# DD-HH:MM:SS
#SBATCH --mail-user=isahay@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-avassile

# Prepare Environment
module load python/3.7 gcc/9.3.0 arrow/2.0.0 cuda cudnn
source config-env.sh

source $ENVDIR/bin/activate && \
echo && echo "LOG STATUS: Activated environment ""$NAME"

export LOGSDIR="$WORKSPACE/logs_$SLURM_JOB_ID" && \
mkdir -p $LOGSDIR

# User defined variables
TASK_PATH="2024/VALIDATION/SNB6_cn_v1/06_June/SNB_20200626"
IMAGE=~/projects/ctb-ruthjoy/CormorantNestingBC/2020-SNB-TIFs/SNB_26062020_71316x34752.tif
IMAGE=$(realpath $IMAGE)
TILE_SIZE=1000
# TRAINED_MODEL=../../../../exported_models/snb6/centernet_resnet101_512/v1/
TRAINED_MODEL=$WORKSPACE/exported_models/snb6/centernet_resnet101_512/v1/

REPO=cormorants-nesting-scripts
PIPELINE=$WORKSPACE/$REPO/object_detection_scripts

# Move to the correct starting point
echo "LOG STATUS: Running pipeline"
# cd cormorants-nesting-scripts/object_detection_scripts/
cd $PIPELINE

# Tile Image 
echo "LOG STATUS: Tiling Image..."
cd $PIPELINE/tile_tifs
python src/tile_tif.py --in_file "$IMAGE" --out_dir output/$TASK_PATH/ --tile_height $TILE_SIZE --tile_width $TILE_SIZE
echo "LOG STATUS: Completed tile_tfs"

# Run Model
echo "LOG STATUS: Predicting..."
cd $PIPELINE/predict
python src/predict.py --tiles $PIPELINE/tile_tifs/output/$TASK_PATH --exported_model $TRAINED_MODEL --out_dir output/$TASK_PATH/ --box_thresh 0.1
echo "LOG STATUS: Completed predictions"

# Post Process Model Results
# cd $PIPELINE/post_process_detections
# python3 $PIPELINE/post_process.py --detections_file $PIPELINE/predict/src/output/$TASK_PATH/detections.csv --out_file $PIPELINE/output/$TASK_PATH/detections_pp2.csv --mask_file $PIPELINE/input/$TASK_PATH/mask.csv

# Remove Temporary Files
rm $PIPELINE/tile_tifs/output/$TASK_PATH/*jpg

# Write Script to Output
echo "LOG STATUS: Saving output and logs"
# cat ../../../../JA_run_model_on_new_pano.sh > ../output/$TASK_PATH/script.bak.log
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

echo "LOG STATUS: Additional logs may be written to $LOGSDIR" & echo
