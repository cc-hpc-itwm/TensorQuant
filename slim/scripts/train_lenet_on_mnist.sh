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
TRAIN_DIR=./tmp/lenet-model/test
rm -r ${TRAIN_DIR}
mkdir ${TRAIN_DIR}

# Where the dataset is saved to.
DATASET_DIR=~/tmp/mnist

# number of total steps and validation interval
STEPS=1000
VAL_INTERVAL=1000

# Download the dataset
#python download_and_convert_data.py \
#  --dataset_name=mnist \
#  --dataset_dir=${DATASET_DIR}

for (( step=VAL_INTERVAL; step<=STEPS; step+=VAL_INTERVAL ))
do 

    # Run training.
    export CUDA_VISIBLE_DEVICES=0
    python train_image_classifier.py \
      --train_dir=${TRAIN_DIR} \
      --dataset_name=mnist \
      --dataset_split_name=train \
      --dataset_dir=${DATASET_DIR} \
      --model_name=lenet \
      --preprocessing_name=lenet \
      --max_number_of_steps=$step \
      --batch_size=128 \
      --learning_rate=0.01 \
      --save_interval_secs=3600 \
      --save_summaries_secs=3600 \
      --log_every_n_steps=100 \
      --optimizer=lars \
      --learning_rate_decay_type=polynomial \
      --end_learning_rate=0.0 \
      --intr_grad_quantizer=logarithmic \
    #  --intr_grad_quantizer=nearest,4,3 \
    #  --intr_grad_quantizer=logarithmic \
    #  --intr_grad_quantizer=nearest,4,3 \
    
    #  --weight_qmap=./tmp/lenet-model/QMaps/weights.json \

    # Run evaluation.
    python eval_image_classifier.py \
      --checkpoint_path=${TRAIN_DIR} \
      --eval_dir=${TRAIN_DIR} \
      --dataset_name=mnist \
      --dataset_split_name=test \
      --dataset_dir=${DATASET_DIR} \
      --model_name=lenet

done
