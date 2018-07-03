#!/bin/bash
#

MODEL=resnet_v1_14

# Where the model is at. checkpoint and logs will be saved to TRAIN_DIR
TOP_DIR=./tmp/resnetv1_14-model


# Name of Dataset
DATASET_NAME=cifar100

# Where the dataset is saved to.
DATASET_DIR=~/tmp/cifar100

# Name of the Experiment
#EXPERIMENT=${MODEL}_train_sparse_grad
#EXP_FOLDER=experiment_results
#EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json

# Passed Parameters
THRESHOLDS="0.1
0.05
0.01
0.001
0.0001
0.00005
0.00001"




# simulating and writing json file
#exec > ${EXP_FILE}
echo "PID:" $$
#exec >> ${EXP_FILE}
echo "######################"
echo "Start Experiment" ${EXPERIMENT}
date
echo "######################"
echo ""
for thresh in ${THRESHOLDS}
do
    TRAIN_DIR=${TOP_DIR}/sparse_grad_thresh_${thresh}

    python train_image_classifier.py \
      --train_dir=${TRAIN_DIR} \
      --dataset_name=${DATASET_NAME} \
      --dataset_split_name=train \
      --dataset_dir=${DATASET_DIR} \
      --model_name=${MODEL} \
      --labels_offset=0 \
      --preprocessing_name=cifarnet \
      --train_image_size=32 \
      --max_number_of_steps=450000 \
      --batch_size=32 \
      --save_interval_secs=3600 \
      --save_summaries_secs=30 \
      --log_every_n_steps=100 \
      --optimizer=momentum \
      --learning_rate=0.1 \
      --weight_decay=0.00004 \
      --extr_grad_quantizer="sparse,${thresh}"


    # Run evaluation.
    python eval_image_classifier.py \
      --checkpoint_path=${TRAIN_DIR} \
      --eval_dir=${TRAIN_DIR} \
      --dataset_name=${DATASET_NAME} \
      --dataset_split_name=test \
      --dataset_dir=${DATASET_DIR} \
      --model_name=${MODEL} \
      --labels_offset=0 \
      --preprocessing_name=cifarnet \
      --eval_image_size=32 \
      --max_num_batches=-1 \
      --batch_size=128
done

echo "######################"
echo "Finished Experiment"
date
echo "######################"
