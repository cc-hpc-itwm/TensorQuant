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

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=mnist \
  --dataset_dir=${DATASET_DIR}

export CUDA_VISIBLE_DEVICES=0
# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=mnist \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet \
  --max_num_batches=100 \
  --batch_size=10 \
  --intr_quantizer=16,8,nearest \
  --intr_quantize_layers= \
  --extr_quantizer=16,8 \
  --extr_quantize_layers= 
