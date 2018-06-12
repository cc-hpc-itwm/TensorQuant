#!/bin/bash
#
# Start script from slim/ directory!

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/resnetv1_50-model/cifar10
rm -r ${TRAIN_DIR}
mkdir ${TRAIN_DIR}

#QMAP=${TRAIN_DIR}/QMaps/optimal_sparse.json

# Name of Dataset
#DATASET_NAME=imagenet
DATASET_NAME=cifar10

# Where the dataset is saved to.
#DATASET_DIR=/data/tf
DATASET_DIR=~/tmp/cifar10


export CUDA_VISIBLE_DEVICES=0

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50 \
  --labels_offset=0 \
  --preprocessing_name=cifarnet \
  --train_image_size=32 \
  --max_number_of_steps=10000 \
  --batch_size=128 \
  --save_interval_secs=600 \
  --save_summaries_secs=600 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --learning_rate=0.01 \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --preprocessing_name=cifarnet \
  --eval_image_size=32 \
  --model_name=resnet_v1_50 \
  --labels_offset=0 \
  --max_num_batches=100 \
  --batch_size=128

unset CUDA_VISIBLE_DEVICES
