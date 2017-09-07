#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the MNIST dataset
# 2. Trains a LeNet model on the MNIST training set.
# 3. Evaluates the model on the MNIST testing set.
#
# Usage:
# cd slim
# ./slim/scripts/train_lenet_on_mnist.sh

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/vgg19-model

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/mnt/beegfs/tf

# Download the dataset
#python download_and_convert_data.py \
#  --dataset_name=${DATASET_NAME} \
#  --dataset_dir=${DATASET_DIR}

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_19 \
  --preprocessing_name=vgg_19 \
  --max_number_of_steps=1 \
  --batch_size=50 \
  --learning_rate=0.01 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --learning_rate_decay_type=fixed \
  --weight_decay=0
