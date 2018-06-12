#!/bin/bash
#
# Dominik Loroch
# 
# Experiment: Explore batch_norm quantization with fixed point.
# 
# The intrinsic quantization of every batch_norm layer is swept.

MODEL=lenet

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/lenet-model
QMAP_DIR=${TRAIN_DIR}/QMaps
QMAP_FILE=weights_tmp.json

# Name of Dataset
DATASET_NAME=mnist

# Where the dataset is saved to.
DATASET_DIR=~/tmp/mnist

# number of validation samples
BATCHES=-1

# Name of the Experiment
EXPERIMENT=lenet_sparsity
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}_$$.json

# input parameters
decades="0.0001"
thresholds=""
for dec in $decades
do
    for i in $(seq 1 2 9)
    do
        thresholds+=" `echo "$dec*$i"|bc -l`"
    done
done

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
python eval_image_classifier.py \
          --checkpoint_path=${TRAIN_DIR} \
          --eval_dir=/tmp/tf \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=test \
          --dataset_dir=${DATASET_DIR} \
          --model_name=${MODEL} \
          --batch_size=10 \
          --max_num_batches=${BATCHES} \
          --output_file=${EXP_FILE}

for threshold in $thresholds
do
    cp ${QMAP_DIR}/weights_template.json ${QMAP_DIR}/${QMAP_FILE}
    sed -i -e "s/#w01/${threshold}/g" ${QMAP_DIR}/${QMAP_FILE}
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
          --weight_qmap=${QMAP_DIR}/${QMAP_FILE} \
          --comment="type=sparse, threshold=${threshold}"
done

echo "######################"
echo "Finished Experiment"
date
echo "######################"
