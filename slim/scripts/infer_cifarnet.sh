#!/bin/bash


# which dataset to use
DATASET_NAME=$1

# Choose Parameters according to used dataset
BASE_DIR=./tmp/cifarnet-model

if [ "$DATASET_NAME" == "cifar10" ]; then
    TRAIN_DIR=${BASE_DIR}/cifar10
    DATASET_DIR=~/tmp/cifar10
    LABELS_OFFSET=0
    IMG_SIZE=32
    PREPROCESSING_NAME=cifarnet
    TRAIN_SPLIT_NAME=train
    TEST_SPLIT_NAME=test
elif [ "$DATASET_NAME" == "cifar100" ]; then
    TRAIN_DIR=${BASE_DIR}/cifar100
    DATASET_DIR=~/tmp/cifar100
    LABELS_OFFSET=0
    IMG_SIZE=32
    PREPROCESSING_NAME=cifarnet
    TRAIN_SPLIT_NAME=train
    TEST_SPLIT_NAME=test
fi

#export CUDA_VISIBLE_DEVICES=0

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${TEST_SPLIT_NAME} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet \
  --max_num_batches=-1 \
  --batch_size=128
