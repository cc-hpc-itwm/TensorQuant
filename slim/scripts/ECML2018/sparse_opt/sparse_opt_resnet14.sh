#!/bin/bash
#
# Start script from slim/ directory!

# Where the checkpoints are at
BASE_DIR=./tmp/resnetv1_14-model
TRAIN_DIR=${BASE_DIR}/l1-regularizer

# A .json file with a list of the layers to be quantized
LAYERS=${BASE_DIR}/layers.json
# A temporary file location, actual location does not matter.
QMAP=/tmp/tf/tmp_qmap.json

# Where the dataset is saved to.
DATASET_DIR=~/tmp/cifar100
# Name of the dataset
DATASET_NAME=cifar100

# extr or weight optimization
OPTIMIZER_MODE=${1}

# Name of the Experiment
EXPERIMENT=resnet14_sparse_opt_${OPTIMIZER_MODE}_$$
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json

python sparse_opt.py \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=test \
    --dataset_dir=${DATASET_DIR} \
    --model_name=resnet_v1_14 \
    --labels_offset=0 \
    --preprocessing_name=cifarnet \
    --eval_image_size=32 \
    --batch_size=128 \
    --max_num_batches=8 \
    --layers_file=${LAYERS} \
    --tmp_qmap=${QMAP} \
    --data_file=${EXP_FILE} \
    --optimizer_init="sparse,1" \
    --optimizer_mode=${OPTIMIZER_MODE} \
    --margin=1.0 \
    --opt_qmap=${TRAIN_DIR}/opt_${OPTIMIZER_MODE}.json


