#!/bin/bash
#
# Start this script from the slim/ directory!

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/inceptionv1-model

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/data/tf

# QMaps
QMAP_DIR=${TRAIN_DIR}/QMaps
WEIGHT_QMAP=${QMAP_DIR}/optimal_sparse_weight.json
EXTR_QMAP=${QMAP_DIR}/optimal_sparse_extr.json

# Name of the Experiment
EXPERIMENT=inceptionv1_sparsity
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}_$$.json

# Number of batches
NUM_BATCHES=-1

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v1 \
  --batch_size=128 \
  --max_num_batches=${NUM_BATCHES} \
  --output_file=${EXP_FILE} \
  --comment="type=baseline"

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v1 \
  --batch_size=128 \
  --max_num_batches=${NUM_BATCHES} \
  --weight_qmap=${WEIGHT_QMAP} \
  --output_file=${EXP_FILE} \
  --comment="type=weight"

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v1 \
  --batch_size=128 \
  --max_num_batches=${NUM_BATCHES} \
  --extr_qmap=${EXTR_QMAP} \
  --output_file=${EXP_FILE} \
  --comment="type=extr"
