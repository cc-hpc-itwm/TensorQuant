#!/bin/bash
#

# Passed Parameters
THRESHOLD=${1}

MODEL=alexnet_v2

# Where the model is at. checkpoint and logs will be saved to TRAIN_DIR
TOP_DIR=./tmp/alexnet-model
TRAIN_DIR=${TOP_DIR}/sparse_extrWeight_thresh_${THRESHOLD}
QMAP_DIR=${TOP_DIR}/QMaps
QMAP_FILE=weights_tmp.json

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/data/tf

# Name of the Experiment
EXPERIMENT=${MODEL}_train_sparse_extrWeight
EXP_FOLDER=experiment_results
EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json



# GPUs to train on
#export CUDA_VISIBLE_DEVICES=1
CLONES=1

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
    cp ${QMAP_DIR}/weights_template.json ${QMAP_DIR}/${QMAP_FILE}
    sed -i -e "s/#w01/${THRESHOLD}/g" ${QMAP_DIR}/${QMAP_FILE}
    # Run training.
    echo "Training: sparsity ${THRESHOLD}"
    python train_image_classifier.py \
      --train_dir=${TRAIN_DIR} \
      --dataset_name=${DATASET_NAME} \
      --dataset_split_name=train \
      --dataset_dir=${DATASET_DIR} \
      --model_name=alexnet_v2 \
      --preprocessing_name=alexnet_v2 \
      --max_number_of_steps=450000 \
      --batch_size=128 \
      --num_clones=${CLONES} \
      --num_preprocessing_threads=16 \
      --save_interval_secs=3600 \
      --save_summaries_secs=3600 \
      --log_every_n_steps=1000 \
      --weight_qmap=${QMAP_DIR}/${QMAP_FILE} \
      --extr_qmap=${QMAP_DIR}/${QMAP_FILE} \

    # Run evaluation
    python eval_image_classifier.py \
        --checkpoint_path=${TRAIN_DIR} \
          --eval_dir=/tmp/tf \
          --dataset_name=${DATASET_NAME} \
          --dataset_split_name=validation \
          --dataset_dir=${DATASET_DIR} \
          --model_name=alexnet_v2 \
          --batch_size=128 \
          --output_file=${EXP_FILE} \
          --weight_qmap=${QMAP_DIR}/${QMAP_FILE} \
          --extr_qmap=${QMAP_DIR}/${QMAP_FILE} \
          --comment="type=sparse, threshold=${THRESHOLD}"

echo ']' >> ${EXP_FILE}

echo "######################"
echo "Finished Experiment"
date
echo "######################"
