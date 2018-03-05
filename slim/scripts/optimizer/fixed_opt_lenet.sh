#!/bin/bash
#
# Start script from slim/ directory!

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/lenet-model

LAYERS=${TRAIN_DIR}/layers.json
QMAP=/tmp/tf/tmp_qmap.json

# Where the dataset is saved to.
DATASET_DIR=~/tmp/mnist

# Name of the Experiment
EXPERIMENT=lenet_opt_$$
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json


export CUDA_VISIBLE_DEVICES=0

python fixed_opt.py \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_name=mnist \
    --dataset_split_name=test \
    --dataset_dir=${DATASET_DIR} \
    --model_name=lenet \
    --batch_size=10 \
    --max_num_batches=5 \
    --layers_file=${LAYERS} \
    --tmp_qmap=${QMAP} \
    --data_file=${EXP_FILE} \
    --optimizer_init="nearest,4,2" \
    --margin=0.99


