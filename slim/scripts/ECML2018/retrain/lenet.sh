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
MODEL_DIR=./tmp/lenet-model

# Directory of the QMaps
QMAP_DIR=${MODEL_DIR}/QMaps

# Where the dataset is saved to.
DATASET_DIR=~/tmp/mnist

# directory in which training is happening
TRAIN_DIR=${MODEL_DIR}/retrain_sparse
rm -r ${TRAIN_DIR}
cp -r ${MODEL_DIR}/baseline $TRAIN_DIR
basestep=1000
 
# Run training.
# export CUDA_VISIBLE_DEVICES=0
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=mnist \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet \
  --preprocessing_name=lenet \
  --max_number_of_steps=$(( $basestep+1000 )) \
  --batch_size=128 \
  --save_interval_secs=3600 \
  --save_summaries_secs=3600 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate=0.01 \
  --weight_qmap=${QMAP_DIR}/optimal_sparse_weight.json

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=mnist \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet


