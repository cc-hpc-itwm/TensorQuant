#!/bin/bash
#
# Start script from slim/ directory!

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/lenet-model

# A .json file with a list of the layers to be quantized
LAYERS=${TRAIN_DIR}/layers.json
# A temporary file location, actual location does not matter.
QMAP=/tmp/tf/tmp_qmap.json

# Where the dataset is saved to.
DATASET_DIR=~/tmp/mnist
DATASET_NAME=mnist

# Name of the Experiment
EXPERIMENT=lenet_sparse_opt_$$
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json

python sparse_opt.py \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=test \
    --dataset_dir=${DATASET_DIR} \
    --model_name=lenet \
    --batch_size=10 \
    --max_num_batches=60 \
    --layers_file=${LAYERS} \
    --tmp_qmap=${QMAP} \
    --data_file=${EXP_FILE} \
    --optimizer_init="sparse,1" \
    --margin=0.99


