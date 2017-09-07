#!/bin/bash
#
# Dominik Loroch
# 
# Experiment: Explore batch_norm quantization with fixed point.
# 
# The intrinsic quantization of every batch_norm layer is swept.

# The used model
MODEL=${1}

# Where the checkpoint is at
TRAIN_DIR=${2}

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=${3}

# number of validation samples
BATCHES=${4}

# list of layers
layers=${5}

# Name of the Experiment
EXPERIMENT=${1}_layerwise
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json


# input parameters
#layers="resnet_v1_50/conv1
#block1
#block2
#block3
#block4
#logits
#"

widths=$(seq 8 2 24)

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
          --max_num_batches=${BATCHES} \
          --output_file=${EXP_FILE}

# simulating and writing json file
for layer_type in $layers
do
for w in $widths
do
    #last_prec=$[$w-1]
    prec=$(seq $[$w/2-2] 2 $[$w/2+2])
    #prec+=" $[$w-1]"
    for q in $prec
    do
    # intrinsic quantization.
    echo ',' >> ${EXP_FILE}
    python eval_image_classifier.py \
        --checkpoint_path=${TRAIN_DIR} \
          --eval_dir=/tmp/tf \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=validation \
          --dataset_dir=${DATASET_DIR} \
          --model_name=${MODEL} \
          --labels_offset=1 \
          --batch_size=2 \
          --max_num_batches=${BATCHES} \
          --output_file=${EXP_FILE} \
          --intr_quantizer=${w},${q},nearest \
          --intr_quantize_layers=$layer_type
    # extrinsic quantization
    echo ',' >> ${EXP_FILE}
    python eval_image_classifier.py \
        --checkpoint_path=${TRAIN_DIR} \
          --eval_dir=/tmp/tf \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=validation \
          --dataset_dir=${DATASET_DIR} \
          --model_name=${MODEL} \
          --labels_offset=1 \
          --batch_size=2 \
          --max_num_batches=${BATCHES} \
          --output_file=${EXP_FILE} \
          --extr_quantizer=${w},${q},nearest \
          --extr_quantize_layers=$layer_type
    done
done
done

echo ']' >> ${EXP_FILE}



echo "######################"
echo "Finished Experiment"
date
echo "######################"
