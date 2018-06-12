#!/bin/bash
#
# Dominik Loroch
# 
# Experiment: Explore batch_norm quantization with fixed point.
# 
# The intrinsic quantization of every batch_norm layer is swept.

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/resnetv1_50-model

# The used model
MODEL=resnet_v1_50

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/mnt/beegfs/tf

# Name of the Experiment
EXPERIMENT=resnet50_batchnorm
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json

# input parameters
widths=$(seq 16 4 32)


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
          --checkpoint_path=${TRAIN_DIR} \
          --eval_dir=/tmp/tf \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=validation \
          --dataset_dir=${DATASET_DIR} \
          --model_name=${MODEL} \
          --labels_offset=1 \
          --batch_size=2 \
          --max_num_batches=200 \
          --output_file=${EXP_FILE}

for w in $widths
do
    prec=$(seq 0 2 $[$w-1])
    prec+=" $[$w-1]"
    for q in $prec
    do
    # Run evaluation.
    echo ',' >> ${EXP_FILE}
    python eval_image_classifier.py \
        --checkpoint_path=${TRAIN_DIR} \
          --eval_dir=${TRAIN_DIR} \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=validation \
          --dataset_dir=${DATASET_DIR} \
          --model_name=${MODEL} \
          --labels_offset=1 \
          --batch_size=2 \
          --max_num_batches=200 \
          --output_file=${EXP_FILE} \
          --intr_quantizer=${w},${q},nearest \
          --intr_quantize_layers=batch_norm
    done
done
echo ']' >> ${EXP_FILE}



echo "######################"
echo "Finished Experiment"
date
echo "######################"
