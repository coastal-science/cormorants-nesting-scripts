#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1  		# Look at Cluster docs for CPU/GPU ratio 
#SBATCH --mem=16G       		# Memory proportional to GPUs: 32000 Cedar
#SBATCH --time=0-1:00:00     		# DD-HH:MM:SS
#SBATCH --mail-user=isahay@sfu.ca
#SBATCH --mail-type=ALL

# Prepare Environment
module load python/3.7

#pip install geopandas # For Post-Processing
source ../tensorflow/bin/activate
#source ../corm_predict_env/bin/activate
module load python/3.7 protobuf gcc/9.3.0 arrow/2.0.0 cuda cudnn
# pip install tensorflow protobuf Cython pycocotools --no-index
# pip install tf-models-official==2.5.1

cd ../models/research
# python -m pip install --user .
cd ../../workspace
# pip install numpy --upgrade 
# pip install pandas==1.3 --no-index

# User defined variables
TASK_PATH="2023/SNB_Span2-Ishan/01_Jan/SNB_2023-01-05/"
# IMAGE="../../../../../../../CormorantNestingBC/2022-06-04_SNB_Span_2_Panorama.tif"
IMAGE="../../../../../../../CormorantNestingBC/2021-05-05_Marina_Panoramas.tif"
TILE_SIZE=1000
TRAINED_MODEL=../../../../exported_models/snb3/centernet_resnet101_512/v1/
# TRAINED_MODEL=../../../../models/snb6/centernet_resnet101_512/vIS/
MASK_FILEPATH=

# Move to the correct starting point
cd cormorants-nesting-scripts/object_detection_scripts/

# Tile Image 
cd tile_tifs/src
python3 tile_tif.py --in_file "$IMAGE" --out_dir ../output/$TASK_PATH/ --tile_height $TILE_SIZE --tile_width $TILE_SIZE

# Run Model
cd ../../predict/src/
python3.7 predict.py --tiles ../../tile_tifs/output/$TASK_PATH --exported_model $TRAINED_MODEL --out_dir ../output/$TASK_PATH/ --box_thresh 0.1

# Post Process Model Results
#cd ../../post_process_detections/src/
#python3 post_process.py --detections_file ../../predict/src/output/$TASK_PATH/detections.csv --out_file ../output/$TASK_PATH/detections_pp3.cs --mask_file ../input/$TASK_PATH/mask.csv

# Write Script to Output
cat ../../../../run_model_on_new_pano.sh > ../output/$TASK_PATH/script.log
