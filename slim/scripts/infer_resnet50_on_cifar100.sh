#!/bin/bash
#
# Start script from slim/ directory!

# Where the checkpoint and logs will be saved to.
#TRAIN_DIR=./tmp/resnetv1_50-model/cifar100
TRAIN_DIR=./tmp/resnetv1_50-model/sparse_grad_thresh_0.000001

#QMAP=${TRAIN_DIR}/QMaps/optimal_sparse.json

# Name of Dataset
#DATASET_NAME=imagenet
DATASET_NAME=cifar100

# Where the dataset is saved to.
#DATASET_DIR=/data/tf
DATASET_DIR=~/tmp/cifar100


export CUDA_VISIBLE_DEVICES=0

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
  --max_num_batches=-1 \
  --batch_size=128 \
#  --comment="baseline" \
#  --output_file="resnetv1_50-cifar100.json"

unset CUDA_VISIBLE_DEVICES