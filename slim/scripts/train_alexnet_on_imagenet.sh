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
TRAIN_DIR=./tmp/alexnet-model/test

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/mnt/beegfs/tf

# Download the dataset
#python download_and_convert_data.py \
#  --dataset_name=${DATASET_NAME} \
#  --dataset_dir=${DATASET_DIR}
echo "######################"
echo "Start training"
date
echo "######################"
# Run training.
export CUDA_VISIBLE_DEVICES=0,1

python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=alexnet_v2 \
  --preprocessing_name=alexnet_v2 \
  --max_number_of_steps=1000 \
  --batch_size=128 \
  --num_clones=2 \
  --num_preprocessing_threads=16 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=1000 

echo "######################"
echo "End training"
date
echo "######################"

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=alexnet_v2 \
  --max_num_batches=100
