#!/bin/bash

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/lenetv2-model

# Where the dataset is saved to.
DATASET_DIR=~/tmp/mnist

# Run training.
#export CUDA_VISIBLE_DEVICES=0
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=mnist \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet_v2 \
  --preprocessing_name=lenet_v2 \
  --max_number_of_steps=3000 \
  --batch_size=128 \
  --learning_rate=0.01 \
  --save_interval_secs=3600 \
  --save_summaries_secs=5 \
  --log_every_n_steps=100 \
  --optimizer=sgd 

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=mnist \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet_v2
