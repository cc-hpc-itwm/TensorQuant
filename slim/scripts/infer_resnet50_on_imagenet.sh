#!/bin/bash
#
# Start script from slim/ directory!

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/resnetv1_50-model/l1-regularizer

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/data/tf

# Run evaluation.
#export CUDA_VISIBLE_DEVICES=0

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50 \
  --labels_offset=1 \
  --max_num_batches=1 \
  --batch_size=128 \
#  --comment='baseline' \
#  --output_file='./experiment_results/resnet50_baseline.json' \
#  --weight_qmap=${TRAIN_DIR}/opt_weight.json

