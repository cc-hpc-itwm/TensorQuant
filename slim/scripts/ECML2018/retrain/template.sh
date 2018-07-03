#!/bin/bash
#

# Where the checkpoint and logs will be saved to.
#MODEL_DIR=./tmp/alexnet-model
MODEL_DIR=${1}
MODEL_NAME=${2}

# Name of Dataset
#DATASET_NAME=imagenet
DATASET_NAME=${3}

# Training phase related parameters
STEPS=${4}
BATCH_SIZE=${5}
LEARNING_RATE=${6}
OPTIMIZER=${7}



# Image Preprocessing
PREPROCESSING_NAME=${MODEL_NAME}

# Where the dataset is saved to.
if [ ${DATASET_NAME} = 'imagenet' ]; then
    DATASET_DIR=/data/tf
    DATASET_SPLIT=validation
fi

if [ ${DATASET_NAME} = 'mnist' ]; then
    DATASET_DIR=~/tmp/mnist
    DATASET_SPLIT=test
fi

# directory in which training is happening
TRAIN_DIR=${MODEL_DIR}/retrain_sparse
rm -r ${TRAIN_DIR}
mkdir ${TRAIN_DIR}

 
# activations or weights optimization
OPTIMIZER_MODE="weight" #change also in python script calls! (--weight_qmap=${TRAIN_QMAP})

# QMaps
LAYERS=${MODEL_DIR}/layers.json
QMAP_DIR=${MODEL_DIR}/QMaps
TRAIN_QMAP=${QMAP_DIR}/optimal_sparse_${OPTIMIZER_MODE}.json
RETRAIN_QMAP=${QMAP_DIR}/optimal_sparse_retrained_${OPTIMIZER_MODE}.json

# Name of the Experiment
EXPERIMENT=${MODEL_NAME}_retrain_sparse_${OPTIMIZER_MODE}_$$
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json

#############################
### 1) Baseline
### No modifications
#############################
python eval_image_classifier.py \
  --checkpoint_path=${MODEL_DIR}/baseline \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${DATASET_SPLIT} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --batch_size=128 \
  --max_num_batches=-1 \
  --output_file=${EXP_FILE} \
  --comment="type=baseline" \
#  --weight_qmap=${TRAIN_QMAP} 

###########################################
### 2) Sparsify, no retrain
### Sparsity map, but without fine-tuning
###########################################
python sparse_opt.py \
    --checkpoint_path=${MODEL_DIR}/baseline \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=${DATASET_SPLIT} \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${MODEL_NAME} \
    --batch_size=128 \
    --max_num_batches=8 \
    --layers_file=${LAYERS} \
    --tmp_qmap=tmp_qmap.json \
    --optimizer_init="sparse,1" \
    --optimizer_mode=${OPTIMIZER_MODE} \
    --opt_qmap=${TRAIN_QMAP} \
    --margin=1.0 \
    --data_file=${MODEL_DIR}/optimizer.json \

python eval_image_classifier.py \
  --checkpoint_path=${MODEL_DIR}/baseline \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${DATASET_SPLIT} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --batch_size=128 \
  --max_num_batches=-1 \
  --output_file=${EXP_FILE} \
  --comment="type=sparse_baseline" \
  --weight_qmap=${TRAIN_QMAP} 
  
###########################################
### 3) Retrain
### Retrain with sparsity
###########################################
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --checkpoint_path=${MODEL_DIR}/baseline \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --preprocessing_name=${PREPROCESSING_NAME} \
  --max_number_of_steps=${STEPS} \
  --batch_size=${BATCH_SIZE} \
  --save_interval_secs=3600 \
  --save_summaries_secs=3600 \
  --log_every_n_steps=100 \
  --optimizer=${OPTIMIZER} \
  --learning_rate=${LEARNING_RATE} \
  --weight_decay=0.00004 \
  --weight_qmap=${TRAIN_QMAP} 

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${DATASET_SPLIT} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --batch_size=128 \
  --max_num_batches=-1 \
  --output_file=${EXP_FILE} \
  --comment="type=sparse_retrained" \
  --weight_qmap=${TRAIN_QMAP} 

###########################################
### 4) Optimize Retrained Model
### Run optimization over retrained model
###########################################
python sparse_opt.py \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=${DATASET_SPLIT} \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${MODEL_NAME} \
    --batch_size=128 \
    --max_num_batches=8 \
    --layers_file=${LAYERS} \
    --tmp_qmap=tmp_qmap.json \
    --optimizer_init="sparse,1" \
    --optimizer_mode=${OPTIMIZER_MODE} \
    --opt_qmap=${RETRAIN_QMAP} \
    --margin=1.0 \
    --data_file=${MODEL_DIR}/optimizer.json \

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=${DATASET_SPLIT} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --batch_size=128 \
  --max_num_batches=-1 \
  --output_file=${EXP_FILE} \
  --comment="type=newsparse_retrained" \
  --weight_qmap=${RETRAIN_QMAP} 


