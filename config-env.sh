# Folders
######################

export SCRATCH=~/scratch
export PROJECT=~/projects/ctb-ruthjoy/jilliana/Tensorflow
# export PROJECT=/project/6059501/jilliana/Tensorflow # with resolved symlinks

export WORKSPACE=$PROJECT/workspace    # Workspace / home directory of code/gitrepo/etc

# Object detection from https://github.com/tensorflow/models/tree/master/research
export TF_MODEL_GARDEN=$PROJECT    # Location of TensorFlow Model Garden
export TF_OBJ_DET=$TF_MODEL_GARDEN/models/research/object_detection

export TFHUB_CACHE_DIR=$SCRATCH/.cache/tfhub_modules # For tensorflow_hub, currently unused

## virtualenv python environment
# export ENVDIR=/tmp/$RANDOM # Cedar/AllianceCan instructions

export NAME=tf-cormorants
export ENVDIR=$PROJECT/tf-cormorants
export DEPS=$PROJECT/dependencies  # pre-downloaded pip dependencies

export LOGSDIR="$WORKSPACE/logs_$NAME"
mkdir -p $LOGSDIR

echo && echo The environment will be named: "$NAME"
echo The logs are in "$LOGSDIR"
echo 

echo & echo "Complete environment creation in the $PROJECT and activate with $ virtualenv --no-download $ENVDIR && source $ENVDIR/bin/activate"
# virtualenv --no-download $ENVDIR && source $ENVDIR/bin/activate

echo & echo "Remove environment with $ deactive && rm -rf $ENVDIR"
echo "Remove remnant directories with $ rm $LOGSDIR"
# rm -rf $ENVDIR
# rm $LOGSDIR


