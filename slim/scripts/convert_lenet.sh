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
TRAIN_DIR=./tmp/lenet-model

# Where the dataset is saved to.
DATASET_DIR=~/tmp/mnist

# Run evaluation.
python converter.py \
  --checkpoint_path=${TRAIN_DIR} \
  --quantization_method=fixed_point

