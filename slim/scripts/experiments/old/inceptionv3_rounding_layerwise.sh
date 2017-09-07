#!/bin/bash
#
# Dominik Loroch
# 
# Experiment: Is the rounding direction (towards negative, towards zero) important?
# 
# .

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/inceptionv3-model

# The used model
MODEL=inception_v3

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/mnt/beegfs/tf

# Name of the Experiment
EXPERIMENT=inceptionv3_rounding_layerwise
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json

# input parameters
modes="zero
down
nearest
"

layers="Conv2d_1a_3x3
Conv2d_2a_3x3
Conv2d_2b_3x3
MaxPool_3a_3x3
Conv2d_3b_1x1
Conv2d_4a_3x3
MaxPool_5a_3x3
Mixed_5b
Mixed_5c
Mixed_5d
Mixed_6a
Mixed_6b
Mixed_6c
Mixed_6d
Mixed_6e
Mixed_7a
Mixed_7b
Mixed_7c
AuxLogits
PreLogits
Logits
"

widths=$(seq 8 4 24)

# simulating and writing json file
#exec > ${EXP_FILE}
echo "PID:" $$
#exec >> ${EXP_FILE}
echo "######################"
echo "Start Experiment" ${EXPERIMENT}
date
echo "######################"
echo ""


# start .json file
echo '[' > ${EXP_FILE}
# baseline
python eval_image_classifier.py \
          --checkpoint_path=${TRAIN_DIR} \
          --eval_dir=/tmp/tf \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=validation \
          --dataset_dir=${DATASET_DIR} \
          --model_name=${MODEL} \
          --batch_size=2 \
          --max_num_batches=200 \
          --output_file=${EXP_FILE}

for mode_type in $modes
do

for layer in $layers
do
for w in $widths
do
    #prec=$(seq 0 4 $[$w-1])
    #prec+=" $[$w-1]"
    prec=$[$w/2]
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
          --batch_size=2 \
          --max_num_batches=200 \
          --output_file=${EXP_FILE} \
          --intr_quantizer=${w},${q},$mode_type \
          --intr_quantize_layers=$layer 

    done    # q
done    # w
done    # layer
done    # mode_type

#end .json file
echo ']' >> ${EXP_FILE}


#end script
echo "######################"
echo "Finished Experiment"
date
echo "######################"
