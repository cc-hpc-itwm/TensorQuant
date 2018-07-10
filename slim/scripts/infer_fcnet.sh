#!/bin/bash


# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/fcnet-model

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
  --model_name=fcnet 
