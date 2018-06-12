#!/bin/bash
#
# Dominik Loroch
# 
# Experiment: Explore batch_norm quantization with fixed point.
# 
# The intrinsic quantization of every batch_norm layer is swept.

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/inceptionv1-model

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/mnt/beegfs/tf

# Name of the Experiment
EXPERIMENT=googlenet_layerwise
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json

# input parameters
layers="Conv2d_1a_7x7
MaxPool_2a_3x3
Conv2d_2b_1x1
Conv2d_2c_3x3
MaxPool_3a_3x3
Mixed_3b
Mixed_3c
MaxPool_4a_3x3
Mixed_4b
Mixed_4c
Mixed_4d
Mixed_4e
Mixed_4f
MaxPool_5a_2x2
Mixed_5b
Mixed_5c
Logits
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
          --eval_dir=${TRAIN_DIR} \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=validation \
          --dataset_dir=${DATASET_DIR} \
          --model_name=inception_v1 \
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
          --eval_dir=${TRAIN_DIR} \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=validation \
          --dataset_dir=${DATASET_DIR} \
          --model_name=inception_v1 \
          --batch_size=2 \
          --max_num_batches=200 \
          --output_file=${EXP_FILE} \
          --intr_quantizer=${w},${q},nearest \
          --intr_quantize_layers=$layer_type
    # extrinsic quantization
    echo ',' >> ${EXP_FILE}
    python eval_image_classifier.py \
        --checkpoint_path=${TRAIN_DIR} \
          --eval_dir=${TRAIN_DIR} \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=validation \
          --dataset_dir=${DATASET_DIR} \
          --model_name=inception_v1 \
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
