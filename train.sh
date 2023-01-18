#!/bin/bash
#SBATCH --gres=gpu:1       		# Request GPU "generic resources"
#SBATCH --cpus-per-task=2  		# Look at Cluster docs for CPU/GPU ratio 
#SBATCH --mem=32G       		# Memory proportional to GPUs: 32000 Cedar
#SBATCH --time=0-8:00:00     		# DD-HH:MM:SS
#SBATCH --mail-user=isahay@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-avassile

# Prepare Environment
module load python/3.7 gcc/9.3.0 arrow/2.0.0 cuda/11.0 cudnn/8.0.3
source config-env.sh
# source ../tensorflow-scratch/bin/activate
source $ENVDIR/bin/activate && \
echo && echo "LOG STATUS: Activated environment ""$NAME"

export LOGSDIR="$WORKSPACE/logs_$SLURM_JOB_ID" && \
mkdir -p $LOGSDIR

# pip install tensorflow protobuf Cython pycocotools --no-index
# pip install tf-models-official==2.5.1
# pip install pyarrow==2.0.0

# cd ../models/research
# python -m pip install --user .
# cd ../../workspace
# pip install numpy --upgrade 
cd $WORKSPACE

#Choose a Model
#MODELDIR=models/gab1/centernet_hourglass_1024/v1
#iMODELDIR=models/gab1/centernet_mobilenet_512/v1
#MODELDIR=models/gab1/centernet_resnet101_512/v1
#MODELDIR=models/gab1/centernet_resnet101_512/v2
#MODELDIR=models/gab2/centernet_resnet101_512/v1
#MODELDIR=models/gab2/centernet_resnet101_512/v2
#MODELDIR=models/gab2/efficientdet_d0/v1 # Gives NaN
#MODELDIR=models/gab2/efficientdet_d0/v2
#MODELDIR=models/gab2/efficientdet_d4/v1
#MODELDIR=models/gab3/centernet_resnet101_512/v1
#Modeldir=models/gab3/centernet_resnet101_512/v2
#MODELDIR=models/gab3/centernet_resnet101_512/v3
#MODELDIR=models/snb5/centernet_resnet101_512/v1
#MODELDIR=models/snb5/centernet_resnet101_512/v2
#MODELDIR=models/snb5/ssd_resnet50_1024/v2
#MODELDIR=models/snb5/ssd_resnet50_1024/v4
MODELDIR=models/snb6/ssd_resnet50_1024/v1
MODELDIR=$WORKSPACE/models/snb6/centernet_resnet101_512/v-IS


# Test the Tensorflow Installation
# python ../models/research/object_detection/builders/model_builder_tf2_test.py
echo && echo "LOG STATUS: Testing Object Detection Model"

# cd $TF_MODEL_GARDEN/models/research
# python object_detection/builders/model_builder_tf2_test.py | tee $LOGSDIR/test-tf-od.log
python $TF_OBJ_DET/builders/model_builder_tf2_test.py #| tee $LOGSDIR/test-tf-od.log

# Tensorboard
echo && echo "LOG STATUS: Launching Tensorboard"
tensorboard --logdir=$MODELDIR --host 0.0.0.0 --load_fast false &

# Starting the Executable
echo && echo "LOG STATUS: Start Training"

python $TF_OBJ_DET/model_main_tf2.py \
  --pipeline_config_path=$MODELDIR/pipeline.config \
  --model_dir=$MODELDIR \
  --checkpoint_every_n=500 \
  --num_workers=1 \
  --alsologtostderr &

# Start the evaluator Script
echo && echo "LOG STATUS: Start Validation"

# export CUDA_VISIBLE_DEVICES=-1
python $TF_OBJ_DET/model_main_tf2.py \
  --pipeline_config_path=$MODELDIR/pipeline.config \
  --model_dir=$MODELDIR \
  --checkpoint_dir=$MODELDIR \
  --num_workers=1 \
  --sample_1_of_n_eval_examples=1

# Write Script to Output
cat train.sh > $MODELDIR/script.bak.log

# Write module list, environment varibles and script to output
full_path=$(realpath $0) # $0 is the name of the current script as it was executed

OUTPUT_FILE=$MODELDIR/script.log
touch $OUTPUT_FILE

echo > $OUTPUT_FILE

section_break(){
  (echo && echo && \
  echo "==============================================" && \
  echo && echo ) >> $MODELDIR/script.log
}
section_break

# get environment variables that contain any of the following keywords
printenv | grep -E 'NAME|ENVDIR|DEPS|LOGSDIR|SCRATCH|PROJECT|WORKSPACE|TF_MODEL_GARDEN|TF_OBJ_DET|TFHUB_CACHE_DIR|MODEL' \
  >> $OUTPUT_FILE

section_break
echo >> $OUTPUT_FILE

cat $full_path >> $OUTPUT_FILE
cp $OUTPUT_FILE $LOGSDIR/ 

echo Additional logs may be written to "$LOGSDIR" & echo
