#!/bin/bash
#
# Start script from slim/ directory!

# Model which should be optimized
MODEL=$1

# optimization subfolder
SUB_MODEL=$2

# extr or weight optimization
OPTIMIZER_MODE=$3

# Where the checkpoint and logs will be saved to.
if [ "$MODEL" == "lenet" ]; then
    BASE_DIR=./tmp/lenet-model
    DATASET_NAME=mnist
    MODEL_NAME=lenet
    LABELS_OFFSET=0
    PREPROCESSING_NAME=lenet
elif [ "$MODEL" == "cifarnet" ]; then
    BASE_DIR=./tmp/cifarnet-model
    DATASET_NAME=cifar10
    MODEL_NAME=cifarnet
    LABELS_OFFSET=0
    PREPROCESSING_NAME=cifarnet
elif [ "$MODEL" == "alexnet" ]; then
    BASE_DIR=./tmp/alexnet-model
    DATASET_NAME=imagenet
    MODEL_NAME=alexnet_v2
    LABELS_OFFSET=0
    PREPROCESSING_NAME=alexnet_v2
elif [ "$MODEL" == "resnet14" ]; then
    BASE_DIR=./tmp/resnetv1_14-model
    DATASET_NAME=cifar100
    MODEL_NAME=resnet_v1_14
    LABELS_OFFSET=0
    PREPROCESSING_NAME=cifarnet
elif [ "$MODEL" == "resnet50" ]; then
    BASE_DIR=./tmp/resnetv1_50-model
    DATASET_NAME=imagenet
    MODEL_NAME=resnet_v1_50
    LABELS_OFFSET=1
    PREPROCESSING_NAME=resnet_v1_50
fi


# Where the dataset is saved to.
if [ "$DATASET_NAME" == "mnist" ]; then
    DATASET_DIR=~/tmp/mnist
    DATASET_TEST_NAME=test
    IMG_SIZE=28
elif [ "$DATASET_NAME" == "imagenet" ]; then
    DATASET_DIR=/data/tf
    DATASET_TEST_NAME=validation
    IMG_SIZE=224
elif [ "$DATASET_NAME" == "cifar100" ]; then
    DATASET_DIR=~/tmp/cifar100
    DATASET_TEST_NAME=test
    IMG_SIZE=32
elif [ "$DATASET_NAME" == "cifar10" ]; then
    DATASET_DIR=~/tmp/cifar10
    DATASET_TEST_NAME=test
    IMG_SIZE=32
fi

# Name of the Experiment
EXPERIMENT=${MODEL}_optimization
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}_$$.json

# Number of batches
NUM_BATCHES=8
BATCH_SIZE=128

TRAIN_DIR=$BASE_DIR/${SUB_MODEL}

python sparse_opt.py \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=${DATASET_TEST_NAME} \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${MODEL_NAME} \
    --labels_offset=${LABELS_OFFSET} \
    --preprocessing_name=${PREPROCESSING_NAME} \
    --eval_image_size=${IMG_SIZE} \
    --batch_size=${BATCH_SIZE} \
    --max_num_batches=${NUM_BATCHES} \
    --layers_file=${BASE_DIR}/layers.json \
    --tmp_qmap=${TRAIN_DIR}/tmp_qmap.json \
    --data_file=${EXP_FILE} \
    --optimizer_init="sparse,1" \
    --optimizer_mode=${OPTIMIZER_MODE} \
    --margin=1.0 \
    --opt_qmap=${TRAIN_DIR}/opt_${OPTIMIZER_MODE}.json


