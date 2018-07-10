#!/bin/bash
#
# Start script from slim/ directory!

# which dataset to use
DATASET_NAME=$1

# Choose Parameters according to used dataset
BASE_DIR=./tmp/resnetv1_14-model

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


# Run training.
#export CUDA_VISIBLE_DEVICES=0

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${TRAIN_SPLIT_NAME} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_14 \
  --train_image_size=${IMG_SIZE} \
  --labels_offset=${LABELS_OFFSET} \
  --preprocessing_name=${PREPROCESSING_NAME} \
  --max_number_of_steps=450000 \
  --batch_size=32 \
  --save_interval_secs=3600 \
  --save_summaries_secs=10 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate=0.1 \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${TEST_SPLIT_NAME} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_14 \
  --labels_offset=${LABELS_OFFSET} \
  --preprocessing_name=${PREPROCESSING_NAME} \
  --eval_image_size=${IMG_SIZE} \
  --max_num_batches=-1 \
  --batch_size=128
  
