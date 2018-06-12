#!/bin/bash
#
# Start this script from the slim/ directory!

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/alexnet-model
#TRAIN_DIR=./tmp/alexnet-model/sparse_extr_0.1
#TRAIN_DIR=./tmp/alexnet-model/logarithmic_long_training

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/data/tf

#export CUDA_VISIBLE_DEVICES=0
# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=alexnet_v2 \
  --max_num_batches=100 \
  --batch_size=128 \
#  --weight_qmap=./tmp/alexnet-model/QMaps/optimal_sparse.json \
#  --extr_qmap=./tmp/alexnet-model/QMaps/optimal_sparse.json 

  
