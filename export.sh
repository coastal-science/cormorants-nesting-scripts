#!/bin/bash
#SBATCH --cpus-per-task=1  		# Look at Cluster docs for CPU/GPU ratio 
#SBATCH --mem=32000M       		# Memory proportional to GPUs: 32000 Cedar
#SBATCH --time=0-00:15:00     		# DD-HH:MM:SS
#SBATCH --mail-type=ALL

# Prepare Environment
module load python/3.7 gcc/9.3.0 arrow/2.0.0 cuda/11.0 cudnn/8.0.3
source config-env.sh
# source ../tensorflow-scratch/bin/activate
source ${ENVDIR}/bin/activate && \
echo && echo "LOG STATUS: Activated environment ""$NAME"

# Prepare Environment
#module load python/3.7 protobuf gcc/9.3.0 arrow/2.0.0 cuda cudnn
#source ../tensorflow/bin/activate
#module load python/3.6 protobuf gcc/9.3.0 arrow/2.0.0 cuda cudnn
#pip install tensorflow protobuf Cython pycocotools --no-index
#pip install tf-models-official==2.5.1
#pip install pyarrow==2.0.0
#python -m pip install --user .
#pip install numpy --upgrade 
#echo Environment Prepared

#Choose a Model
#MODELDIR=gab1/centernet_hourglass_1024/v1
#iMODELDIR=gab1/centernet_mobilenet_512/v1
#MODELDIR=gab2/centernet_resnet101_512/v1
#MODELDIR=gab2/centernet_resnet101_512/v2
#MODELDIR=snb1/centernet_resnet101_512/v4
#MODELDIR=snb1/centernet_resnet101_512/v3
#MODELDIR=snb2/centernet_resnet101_512/v1
#MODELDIR=snb3/centernet_resnet101_512/v1
#MODELDIR=gab3/centernet_resnet101_512/v3
MODELDIR=snb5/centernet_resnet101_512/w4
#MODELDIR=snb6/ssd_resnet50_1024/v7
#MODELDIR=snb6/ssd_resnet50_1024/v8
MODELDIR=snb5/centernet_hg104_512/v9

# Export the model
python $TF_OBJ_DET/exporter_main_v2.py \
  --pipeline_config_path=models/$MODELDIR/pipeline.config \
  --trained_checkpoint_dir=models/$MODELDIR/ \
  --output_directory=exported_models/$MODELDIR/ \
  --input_type=image_tensor


