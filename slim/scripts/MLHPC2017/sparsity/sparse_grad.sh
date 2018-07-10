#!/bin/bash
#
# Dominik Loroch
# 
# Experiment: Explore batch_norm quantization with fixed point.
# 
# The intrinsic quantization of every batch_norm layer is swept.

MODEL=${1}

# Where the checkpoints are saved.
CHECKPOINT_DIR=${2}

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved.
DATASET_DIR=${3}

# number of validation samples
BATCHES=${4}

# input parameters
gradients="0.02
0.01
0.008
0.002
0.004
0.001
0.0001
0.00005
0.00001"

# Name of the Experiment
EXPERIMENT=${1}_sparse_grad
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json

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
          --checkpoint_path=${CHECKPOINT_DIR} \
          --eval_dir=/tmp/tf \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=validation \
          --dataset_dir=${DATASET_DIR} \
          --model_name=${MODEL} \
          --batch_size=2 \
          --max_num_batches=${BATCHES} \
          --output_file=${EXP_FILE} \
          --comment='baseline'

for g in $gradients;
do
    echo ',' >> ${EXP_FILE}
    python eval_image_classifier.py \
              --checkpoint_path="${CHECKPOINT_DIR}/sparse_grad_thresh_${g}" \
              --eval_dir=/tmp/tf \
              --dataset_name=${DATASET_NAME} \
              --dataset_split_name=validation \
              --dataset_dir=${DATASET_DIR} \
              --model_name=${MODEL} \
              --batch_size=2 \
              --max_num_batches=${BATCHES} \
              --output_file=${EXP_FILE} \
              --comment="threshold=$g"

done
echo ']' >> ${EXP_FILE}

echo "######################"
echo "Finished Experiment"
date
echo "######################"
