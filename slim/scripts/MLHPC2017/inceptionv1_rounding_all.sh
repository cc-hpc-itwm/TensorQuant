#!/bin/bash
#
# Dominik Loroch
# 
# Experiment: Is the rounding direction (towards negative, towards zero) important?
# 
# .

# The used model
MODEL=inception_v1

# Where the checkpoint is at
TRAIN_DIR=./tmp/inceptionv1-model

# Directory with QMaps
QMAP_DIR=${TRAIN_DIR}/QMaps

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/data/tf

# number of validation samples
BATCHES=60

# Name of the Experiment
EXPERIMENT=${MODEL}_rounding_all_$$
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json

# input parameters
modes="zero
down
nearest
stochastic
"

widths=$(seq 8 4 32)


echo "PID:" $$
echo "######################"
echo "Start Experiment" ${EXPERIMENT}
date
echo "######################"
echo ""

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
          --output_file=${EXP_FILE} \
          --comment="type=baseline"

for mode_type in $modes
do
for w in $widths
do
    prec=$[$w/2]
    for f in $prec
    do
    cp ${QMAP_DIR}/rounding_tmp.json ${QMAP_DIR}/temp.json
    sed -i -e "s/#q/${mode_type}/g" ${QMAP_DIR}/temp.json
    sed -i -e "s/#w/${w}/g" ${QMAP_DIR}/temp.json
    sed -i -e "s/#f/${f}/g" ${QMAP_DIR}/temp.json
    # intrinsic quantization.
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
          --intr_qmap=${QMAP_DIR}/temp.json \
          --comment="type=${mode_type} width=${w} frac=${f}"

    rm ${QMAP_DIR}/temp.json

    done    # q
done    # w
done    # mode_type


#end script
echo "######################"
echo "Finished Experiment"
date
echo "######################"
