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
TRAIN_DIR=./tmp/cifarnet-model/test
rm -r ${TRAIN_DIR}
mkdir ${TRAIN_DIR}

# Where the dataset is saved to.
DATASET_DIR=~/tmp/cifar100

# Run training.
export CUDA_VISIBLE_DEVICES=0

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cifar100 \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet \
  --preprocessing_name=cifarnet \
  --max_number_of_steps=10000 \
  --batch_size=128 \
  --save_interval_secs=3600 \
  --save_summaries_secs=30 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --intr_grad_quantizer='' \
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=200 \
  --weight_decay=0.004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=cifar100 \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet
