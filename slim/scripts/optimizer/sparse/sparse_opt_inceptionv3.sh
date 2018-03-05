#!/bin/bash
#
# Start script from slim/ directory!

# Where the checkpoints are at
TRAIN_DIR=./tmp/inceptionv3-model

# A .json file with a list of the layers to be quantized
LAYERS=${TRAIN_DIR}/layers.json
# A temporary file location, actual location does not matter.
QMAP=/tmp/tf/tmp_qmap.json

# Where the dataset is saved to.
DATASET_DIR=/data/tf
# Name of the dataset
DATASET_NAME=imagenet

# Name of the Experiment
EXPERIMENT=inceptionv3_sparse_opt_$$
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json

python sparse_opt.py \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=validation \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v3 \
    --batch_size=10 \
    --max_num_batches=60 \
    --layers_file=${LAYERS} \
    --tmp_qmap=${QMAP} \
    --data_file=${EXP_FILE} \
    --optimizer_init="sparse,1" \
    --margin=0.99


