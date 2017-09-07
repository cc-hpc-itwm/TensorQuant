#!/bin/bash
#
# Dominik Loroch
# 
# Experiment: Explore batch_norm quantization with fixed point.
# 
# The intrinsic quantization of every batch_norm layer is swept.

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/resnetv1_50-model

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/mnt/beegfs/tf

# The used model
MODEL=resnet_v1_50

# Name of the Experiment
EXPERIMENT=resnet50_layerwise
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json

# input parameters
layers="resnet_v1_50/conv1
block1
block2
block3
block4
logits
"

widths=$(seq 8 4 24)

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

# simulating and writing json file
for layer_type in $layers
do
for w in $widths
do
    #last_prec=$[$w-1]
    prec=$(seq 0 2 $[$w-1])
    prec+=" $[$w-1]"
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
          --max_num_batches=200 \
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
          --max_num_batches=200 \
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
