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

# Run training.
#export CUDA_VISIBLE_DEVICES=0

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${TRAIN_SPLIT_NAME} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet \
  --preprocessing_name=${PREPROCESSING_NAME} \
  --max_number_of_steps=10000 \
  --batch_size=128 \
  --save_interval_secs=3600 \
  --save_summaries_secs=5 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=10 \
  --weight_decay=0.004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${TEST_SPLIT_NAME} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet
