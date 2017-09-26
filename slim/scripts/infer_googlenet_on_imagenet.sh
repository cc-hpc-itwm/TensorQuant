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
TRAIN_DIR=./tmp/inceptionv1-model

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/data/tf

# Run evaluation.
export CUDA_VISIBLE_DEVICES=0
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v1 \
  --max_num_batches=10 \
  --batch_size=2 \
  --intr_quantizer=16,8,nearest \
  --intr_quantize_layers=Conv
#  --intr_quantizer=16,8,nearest \
#  --intr_quantize_layers=conv2d \


unset CUDA_VISIBLE_DEVICES
