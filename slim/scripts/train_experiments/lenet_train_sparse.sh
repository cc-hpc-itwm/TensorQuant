#!/bin/bash
#
# Dominik Loroch
# 
# Experiment: Explore batch_norm quantization with fixed point.
# 
# The intrinsic quantization of every batch_norm layer is swept.

MODEL=lenet

# Where the checkpoint and logs will be saved to.
TOP_DIR=${2}
TRAIN_DIR=${2}/tmp
rm -r ${TRAIN_DIR}
mkdir ${TRAIN_DIR}

# Name of Dataset
DATASET_NAME=mnist

# Where the dataset is saved at.
DATASET_DIR=${3}

# number of validation samples
BATCHES=${4}

# Name of the Experiment
EXPERIMENT=${1}_train_lenet
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json


widths=$(seq 2 2 12)

# Download the dataset
#python download_and_convert_data.py \
#  --dataset_name=mnist \
#  --dataset_dir=${DATASET_DIR}

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
python eval_image_classifier.py \
          --checkpoint_path=${TOP_DIR} \
          --eval_dir=/tmp/tf \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=test \
          --dataset_dir=${DATASET_DIR} \
          --model_name=${MODEL} \
          --batch_size=10 \
          --max_num_batches=${BATCHES} \
          --output_file=${EXP_FILE} \
          --comment="baseline"

for w in $widths
do
    #prec=$(seq 0 4 $[$w-1])
    #prec+=" $[$w-1]"
    prec=$[$w/2]
    for q in $prec
    do
    # quantization.
    echo ',' >> ${EXP_FILE}
    # Run training.
    python train_image_classifier.py \
      --train_dir=${TRAIN_DIR} \
      --dataset_name=${DATASET_NAME} \
      --dataset_split_name=train \
      --dataset_dir=${DATASET_DIR} \
      --model_name=${MODEL} \
      --preprocessing_name=${MODEL} \
      --max_number_of_steps=20000 \
      --batch_size=50 \
      --learning_rate=0.01 \
      --save_interval_secs=60 \
      --save_summaries_secs=60 \
      --log_every_n_steps=100 \
      --optimizer=sgd \
      --learning_rate_decay_type=fixed \
      --weight_decay=0 \
      --extr_grad_quantizer=nearest,$w,$q 

    # Run evaluation
    python eval_image_classifier.py \
        --checkpoint_path=${TRAIN_DIR} \
          --eval_dir=/tmp/tf \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=test \
          --dataset_dir=${DATASET_DIR} \
          --model_name=${MODEL} \
          --batch_size=10 \
          --max_num_batches=${BATCHES} \
          --output_file=${EXP_FILE} \
          --comment="type=nearest, w=$w, q=$q"

    rm -r ${TRAIN_DIR}
    mkdir ${TRAIN_DIR}
    done
done

echo ']' >> ${EXP_FILE}

echo "######################"
echo "Finished Experiment"
date
echo "######################"
