#!/bin/bash
#
# Start script from slim/ directory!

# which dataset to use
DATASET_NAME=$1

# Choose Parameters according to used dataset
BASE_DIR=./tmp/resnetv1_20-model

if [ "$DATASET_NAME" == "cifar100" ]; then
    TRAIN_DIR=${BASE_DIR}/cifar100
    DATASET_DIR=~/tmp/cifar100
fi



#export CUDA_VISIBLE_DEVICES=0

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_20 \
  --labels_offset=0 \
  --preprocessing_name=cifarnet \
  --eval_image_size=32 \
  --max_num_batches=-1 \
  --batch_size=128

