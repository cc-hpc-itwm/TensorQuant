#!/bin/bash
#
# Start script from slim/ directory!

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/lenet-model

# Where the dataset is saved to.
DATASET_DIR=~/tmp/mnist

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=mnist \
  --dataset_dir=${DATASET_DIR}

export CUDA_VISIBLE_DEVICES=0
# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=mnist \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet \
  --max_num_batches=10 \
  --batch_size=10 \
  --comment='Run Description.' \
  --intr_qmap=${TRAIN_DIR}/QMaps/intrinsic.json \
#  --weight_qmap=${TRAIN_DIR}/QMaps/weights.json \ 
#  --extr_qmap=${TRAIN_DIR}/QMaps/extrinsic.json
