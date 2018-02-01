#!/bin/bash
#
# Start script from slim/ directory!

# Where the checkpoints are at
TRAIN_DIR=./tmp/inceptionv1-model

# Where the quantizer maps are at
QMAP_DIR=${TRAIN_DIR}/QMaps     
# A quantizer map with all layers to be optimized and their initial quantizer parameters
INIT_QMAP_FILE=${QMAP_DIR}/opt_init.json  
# A temporary file 
INTR_QMAP_FILE=${QMAP_DIR}/intr_qmap.json

# Where the dataset is saved to.
DATASET_DIR=/data/tf
# Name of the dataset
DATASET_NAME=imagenet

# Name of the Experiment
EXPERIMENT=googlenet_opt_${MARGIN}_$$
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json


export CUDA_VISIBLE_DEVICES=0

python fixed_opt.py \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=validation \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v1 \
    --batch_size=10 \
    --max_num_batches=1 \
    --init_qmap=${INIT_QMAP_FILE} \
    --intr_qmap=${INTR_QMAP_FILE} \
    --data_file=${EXP_FILE}


