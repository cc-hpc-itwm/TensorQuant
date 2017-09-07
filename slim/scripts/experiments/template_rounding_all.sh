#!/bin/bash
#
# Dominik Loroch
# 
# Experiment: Is the rounding direction (towards negative, towards zero) important?
# 
# .

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

# Name of the Experiment
EXPERIMENT=${1}_rounding_all
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json

# input parameters
modes="zero
down
nearest
"

layers="conv2d,max_pool2d,avg_pool2d,batch_norm
"

widths=$(seq 8 4 32)

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
          --max_num_batches=${BATCHES} \
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
          --max_num_batches=${BATCHES} \
          --output_file=${EXP_FILE} \
          --intr_quantizer=${w},${q},$mode_type \
          --intr_quantize_layers=$layer 

    # extrinsic quantization
    echo ',' >> ${EXP_FILE}
    python eval_image_classifier.py \
        --checkpoint_path=${TRAIN_DIR} \
          --eval_dir=/tmp/tf \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=validation \
          --dataset_dir=${DATASET_DIR} \
          --model_name=${MODEL} \
          --batch_size=2 \
          --max_num_batches=${BATCHES} \
          --output_file=${EXP_FILE} \
          --extr_quantizer=${w},${q},$mode_type \
          --extr_quantize_layers=$layer 

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
