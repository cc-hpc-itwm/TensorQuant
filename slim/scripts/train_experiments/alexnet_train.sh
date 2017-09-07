#!/bin/bash
#
# Dominik Loroch
# 
# Experiment: Explore batch_norm quantization with fixed point.
# 
# The intrinsic quantization of every batch_norm layer is swept.

MODEL=alexnet_v2

# Where the checkpoint and logs will be saved to.
TOP_DIR=./tmp/alexnet-model


# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/mnt/beegfs/tf

# Name of the Experiment
EXPERIMENT=${MODEL}_train
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json


# GPUs to train on
export CUDA_VISIBLE_DEVICES=0,1
CLONES=2


widths="16"

# simulating and writing json file
#exec > ${EXP_FILE}
echo "PID:" $$
#exec >> ${EXP_FILE}
echo "######################"
echo "Start Experiment" ${EXPERIMENT}
date
echo "######################"
echo ""

# baseline
echo '[' > ${EXP_FILE}
#python eval_image_classifier.py \
#          --checkpoint_path=${TOP_DIR} \
#          --eval_dir=/tmp/tf \
#          --dataset_name=${DATASET_NAME} \
#          --dataset_split_name=validation \
#         --dataset_dir=${DATASET_DIR} \
#          --model_name=${MODEL} \
#          --batch_size=2 \
#          --max_num_batches=${BATCHES} \
#          --output_file=${EXP_FILE}

for w in $widths
do
    
    #prec=$(seq 0 4 $[$w-1])
    #prec+=" $[$w-1]"
    #prec=$[$w/2]
    prec="12"
    for q in $prec
    do
    TRAIN_DIR=${TOP_DIR}/width_${w}_${q}

    # training.
    echo ',' >> ${EXP_FILE}
    # Run training.
    echo "Training: ${w},${q}"
    python train_image_classifier.py \
      --train_dir=${TRAIN_DIR} \
      --dataset_name=${DATASET_NAME} \
      --dataset_split_name=train \
      --dataset_dir=${DATASET_DIR} \
      --model_name=alexnet_v2 \
      --preprocessing_name=alexnet_v2 \
      --max_number_of_steps=450000 \
      --batch_size=128 \
      --num_clones=${CLONES} \
      --num_preprocessing_threads=16 \
      --save_interval_secs=3600 \
      --save_summaries_secs=3600 \
      --log_every_n_steps=1000 \
      --grad_quantizer=$w,$q,nearest

    # Run evaluation
    python eval_image_classifier.py \
        --checkpoint_path=${TRAIN_DIR} \
          --eval_dir=/tmp/tf \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=validation \
          --dataset_dir=${DATASET_DIR} \
          --model_name=alexnet_v2 \
          --batch_size=128 \
          --intr_quantizer=$w,$q \
          --output_file=${EXP_FILE}
    
    done
done

echo ']' >> ${EXP_FILE}

echo "######################"
echo "Finished Experiment"
date
echo "######################"
