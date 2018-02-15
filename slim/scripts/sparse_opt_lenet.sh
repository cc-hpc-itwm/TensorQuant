#!/bin/bash
#
# Start script from slim/ directory!

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/lenet-model

QMAP_DIR=${TRAIN_DIR}/QMaps
INIT_QMAP_FILE=${QMAP_DIR}/opt_sparse_init.json
WEIGHT_QMAP_FILE=${QMAP_DIR}/weight_qmap.json

# Where the dataset is saved to.
DATASET_DIR=~/tmp/mnist

# Name of the Experiment
EXPERIMENT=lenet_sparse_opt_$$
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json


export CUDA_VISIBLE_DEVICES=0

python sparse_opt.py \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_name=mnist \
    --dataset_split_name=test \
    --dataset_dir=${DATASET_DIR} \
    --model_name=lenet \
    --batch_size=10 \
    --max_num_batches=5 \
    --init_qmap=${INIT_QMAP_FILE} \
    --weight_qmap=${WEIGHT_QMAP_FILE} \
    --data_file=${EXP_FILE} \
    --margin=0.99


