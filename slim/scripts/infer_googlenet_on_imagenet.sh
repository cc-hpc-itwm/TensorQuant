#!/bin/bash
#
# Start this script from the slim/ directory!

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/inceptionv1-model

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
  --model_name=inception_v1 \
  --max_num_batches=60 \
  --batch_size=10 \
  --weight_qmap=${TRAIN_DIR}/QMaps/weights.json \
  --extr_qmap=${TRAIN_DIR}/QMaps/weights.json

