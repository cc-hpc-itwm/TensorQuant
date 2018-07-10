#!/bin/bash
#
# Start script from slim/ directory!

# which dataset to use
DATASET_NAME=$1

# Choose Parameters according to used dataset
BASE_DIR=./tmp/resnetv1_50-model

if [ "$DATASET_NAME" == "imagenet" ]; then
    TRAIN_DIR=${BASE_DIR}
    DATASET_DIR=/data/tf
    LABELS_OFFSET=1
    IMG_SIZE=224
    PREPROCESSING_NAME=resnet_v1
    DATASET_SPLIT_NAME=validation
fi

#export CUDA_VISIBLE_DEVICES=0

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${DATASET_SPLIT_NAME} \
  --dataset_dir=${DATASET_DIR} \
  --preprocessing_name=${PREPROCESSING_NAME} \
  --eval_image_size=${IMG_SIZE} \
  --model_name=resnet_v1_50 \
  --labels_offset=${LABELS_OFFSET} \
  --max_num_batches=10 \
  --batch_size=32

