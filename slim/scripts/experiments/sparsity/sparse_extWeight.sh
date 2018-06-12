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
QMAP_DIR=${CHECKPOINT_DIR}/QMaps
QMAP_FILE=weights_tmp.json

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved.
DATASET_DIR=${3}

# number of validation samples
BATCHES=${4}

# input parameters
extrWeights="0.0"
#"0.1
#0.08
#0.05
#0.02
#0.01
#0.005
#0.001"

sweepThresh=" 0.5
0.1
0.05
0.045
0.04
0.035
0.03
0.025
0.02
0.015
0.01
0.005
0.001
0.0005
0.0001
"
# input parameters
#decades="0.1 0.01 0.001 0.0001"
#sweepThresh=""
#for dec in $decades
#do
#    for i in $(seq 1 2 9)
#    do
#        sweepThresh+=" `echo "$dec*$i"|bc -l`"
#    done
#done



# Name of the Experiment
EXPERIMENT=${1}_sparse_extrWeight
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}_$$.json

# simulating and writing json file
#exec > ${EXP_FILE}
echo "PID:" $$
#exec >> ${EXP_FILE}
echo "######################"
echo "Start Experiment" ${EXPERIMENT}
date
echo "######################"
echo ""

for g in $extrWeights;
do
# baseline
python eval_image_classifier.py \
          --checkpoint_path="${CHECKPOINT_DIR}/sparse_extrWeight_thresh_${g}" \
          --eval_dir=/tmp/tf \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=validation \
          --dataset_dir=${DATASET_DIR} \
          --model_name=${MODEL} \
          --batch_size=2 \
          --max_num_batches=${BATCHES} \
          --output_file=${EXP_FILE} \
          --comment="baseline threshold=$g"

for s in $sweepThresh;
do

    cp ${QMAP_DIR}/weights_template.json ${QMAP_DIR}/${QMAP_FILE}
    sed -i -e "s/#w01/${s}/g" ${QMAP_DIR}/${QMAP_FILE}
    python eval_image_classifier.py \
              --checkpoint_path="${CHECKPOINT_DIR}/sparse_extrWeight_thresh_${g}" \
              --eval_dir=/tmp/tf \
              --dataset_name=${DATASET_NAME} \
              --dataset_split_name=validation \
              --dataset_dir=${DATASET_DIR} \
              --model_name=${MODEL} \
              --batch_size=2 \
              --max_num_batches=${BATCHES} \
              --output_file=${EXP_FILE} \
              --weight_qmap=${QMAP_DIR}/${QMAP_FILE} \
              --extr_qmap=${QMAP_DIR}/${QMAP_FILE} \
              --comment="train_threshold=$g infer_threshold=$s"

done
done

echo "######################"
echo "Finished Experiment"
date
echo "######################"
