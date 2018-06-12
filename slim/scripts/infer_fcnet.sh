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
TRAIN_DIR=./tmp/fcnet-model/test

# Where the dataset is saved to.
DATASET_DIR=~/tmp/mnist

export CUDA_VISIBLE_DEVICES=0
# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=mnist \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=fcnet \
#  --weight_qmap=${TRAIN_DIR}/QMaps/weights.json
