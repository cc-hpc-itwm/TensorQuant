#!/bin/bash
#

# Where the checkpoint and logs will be saved to.
MODEL_DIR=./tmp/inceptionv1-model

# Directory of the QMaps
QMAP_DIR=${MODEL_DIR}/QMaps

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/data/tf

# directory in which training is happening
TRAIN_DIR=${MODEL_DIR}/retrain_sparse
rm -r ${TRAIN_DIR}
cp -r ${MODEL_DIR}/baseline $TRAIN_DIR
basestep=0
 
# Run training.
# export CUDA_VISIBLE_DEVICES=0
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v1 \
  --preprocessing_name=inception_v1 \
  --max_number_of_steps=$(( $basestep+10000 )) \
  --batch_size=32 \
  --save_interval_secs=600 \
  --save_summaries_secs=600 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --learning_rate=0.01 \
  --weight_decay=0.00004 \
  --ignore_missing_vars=True \
  --checkpoint_exclude_scopes="XXXXX" \
  --weight_qmap=${QMAP_DIR}/optimal_sparse_weight.json

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v1 \
  --batch_size=128 \
  --max_num_batches=400 \
  --weight_qmap=${QMAP_DIR}/optimal_sparse_weight.json


