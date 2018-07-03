#!/bin/bash
#
# Start script from slim/ directory!

MODEL=$1

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
EXPERIMENT=${MODEL}_sparsity
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}_$$.json

# Number of batches
NUM_BATCHES=-1
BATCH_SIZE=128

# L2 Regularizer
TRAIN_DIR=${BASE_DIR}/baseline
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${DATASET_TEST_NAME} \
  --dataset_dir=${DATASET_DIR} \
  --labels_offset=${LABELS_OFFSET} \
  --preprocessing_name=${PREPROCESSING_NAME} \
  --eval_image_size=${IMG_SIZE} \
  --model_name=${MODEL_NAME} \
  --max_num_batches=${NUM_BATCHES} \
  --batch_size=${BATCH_SIZE} \
  --output_file=${EXP_FILE} \
  --comment="type=baseline"

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${DATASET_TEST_NAME} \
  --dataset_dir=${DATASET_DIR} \
  --labels_offset=${LABELS_OFFSET} \
  --preprocessing_name=${PREPROCESSING_NAME} \
  --eval_image_size=${IMG_SIZE} \
  --model_name=${MODEL_NAME} \
  --max_num_batches=${NUM_BATCHES} \
  --batch_size=${BATCH_SIZE} \
  --weight_qmap=${TRAIN_DIR}/opt_weight.json \
  --output_file=${EXP_FILE} \
  --comment="type=L2"

# L1 Regularizer
TRAIN_DIR=${BASE_DIR}/l1-regularizer
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${DATASET_TEST_NAME} \
  --dataset_dir=${DATASET_DIR} \
  --labels_offset=${LABELS_OFFSET} \
  --preprocessing_name=${PREPROCESSING_NAME} \
  --eval_image_size=${IMG_SIZE} \
  --model_name=${MODEL_NAME} \
  --max_num_batches=${NUM_BATCHES} \
  --batch_size=${BATCH_SIZE} \
  --weight_qmap=${TRAIN_DIR}/opt_weight.json \
  --output_file=${EXP_FILE} \
  --comment="type=L1"
