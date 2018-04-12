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
TRAIN_DIR=./tmp/lenet_v2-model/test
rm -r ${TRAIN_DIR}
mkdir ${TRAIN_DIR}

# Where the dataset is saved to.
DATASET_DIR=~/tmp/mnist

# Run training.
export CUDA_VISIBLE_DEVICES=0
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=mnist \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet_v2 \
  --preprocessing_name=lenet_v2 \
  --max_number_of_steps=3000 \
  --batch_size=10 \
  --learning_rate=0.001 \
  --save_interval_secs=3600 \
  --save_summaries_secs=3600 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate_decay_type=fixed \
  --end_learning_rate=0.0001 \
#  --intr_grad_quantizer=nearest,4,3 \
#  --weight_qmap=./tmp/lenet_v2-model/QMaps/weights.json \





# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=mnist \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet_v2 \
#  --weight_qmap=./tmp/lenet_v2-model/QMaps/weights.json \
