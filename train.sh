#!/bin/bash
#SBATCH --gpus-per-node=2     # Request GPU "generic resources"
#SBATCH --cpus-per-task=6   	# Look at Cluster docs for CPU/GPU ratio 
#SBATCH --mem=32000M     		  # Memory proportional to GPUs: 32000 Cedar
#SBATCH --time=0-6:00:00      # DD-HH:MM:SSs
#SBATCH --mail-user=isahay@sfu.ca
#SBATCH --mail-type=ALL

# Load functions
. utils.sh # progress()

# Prepare Environment
module load python/3.7 gcc/9.3.0 arrow/2.0.0 cuda/11.0 cudnn/8.0.3
source config-env.sh
# source ../tensorflow-scratch/bin/activate
source ${ENVDIR}/bin/activate && \
echo && echo "LOG STATUS: Activated environment ""$NAME"

export LOGSDIR="${WORKSPACE}/logs_${SLURM_JOB_ID}" && \
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
#MODELDIR=models/gab1/centernet_mobilenet_512/v1
#MODELDIR=models/gab1/centernet_resnet101_512/v1
#MODELDIR=models/gab1/centernet_resnet101_512/v2
#MODELDIR=models/gab2/centernet_resnet101_512/v1
#MODELDIR=models/gab2/centernet_resnet101_512/v2
#MODELDIR=models/gab2/efficientdet_d0/v1 # Gives NaN
#MODELDIR=models/gab2/efficientdet_d0/v2
#MODELDIR=models/gab2/efficientdet_d4/v1
#MODELDIR=models/gab3/centernet_resnet101_512/v1
#MODELDIR=models/gab3/centernet_resnet101_512/v2
#MODELDIR=models/gab3/centernet_resnet101_512/v3
#MODELDIR=models/snb5/centernet_resnet101_512/v1
#MODELDIR=models/snb5/centernet_resnet101_512/v2
#MODELDIR=models/snb5/ssd_resnet50_1024/v2
#MODELDIR=models/snb5/ssd_resnet50_1024/v4
MODELDIR=$WORKSPACE/models/snb6/ssd_resnet50_1024/v1
MODELDIR=$WORKSPACE/models/snb6/ssd_resnet50_1024/v3-Overfit
MODELDIR=$WORKSPACE/models/snb6/ssd_resnet50_1024/v4
MODELDIR=$WORKSPACE/models/snb6/ssd_resnet50_1024/v5
MODELDIR=$WORKSPACE/models/snb6/ssd_resnet50_1024/v7
MODELDIR=$WORKSPACE/models/snb6/ssd_resnet50_1024/v8-b

# Write Script and Job details to file
progress $MODELDIR $LOGSDIR

# Test the Tensorflow Installation
# python ../models/research/object_detection/builders/model_builder_tf2_test.py
echo && echo "LOG STATUS: Testing Object Detection Model"

# cd $TF_MODEL_GARDEN/models/research
# python object_detection/builders/model_builder_tf2_test.py
python $TF_OBJ_DET/builders/model_builder_tf2_test.py

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

export CUDA_VISIBLE_DEVICES=-1
python $TF_OBJ_DET/model_main_tf2.py \
  --pipeline_config_path=$MODELDIR/pipeline.config \
  --model_dir=$MODELDIR \
  --checkpoint_dir=$MODELDIR \
  --num_workers=1 \
  --sample_1_of_n_eval_examples=1

# Write Script to Output
cat train.sh > $MODELDIR/script.bak.log
