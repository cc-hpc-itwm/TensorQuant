#!/bin/bash
#

MODEL=lenet

# Where the model is at. checkpoint and logs will be saved to TRAIN_DIR
TOP_DIR=./tmp/lenet-model


# Name of Dataset
DATASET_NAME=mnist

# Where the dataset is saved to.
DATASET_DIR=~/tmp/mnist

# Name of the Experiment
EXPERIMENT=${MODEL}_train_sparse_grad
#EXP_FOLDER=experiment_results
#EXP_FILE=./${EXP_FOLDER}/${EXPERIMENT}.json

# Passed Parameters
THRESHOLDS="0.01
0.001
0.0001
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
      --preprocessing_name=lenet \
      --max_number_of_steps=5000 \
      --batch_size=128 \
      --save_interval_secs=3600 \
      --save_summaries_secs=5 \
      --log_every_n_steps=100 \
      --optimizer=sgd \
      --learning_rate=0.01 \
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
      --preprocessing_name=lenet \
      --max_num_batches=-1 \
      --batch_size=128
done

echo "######################"
echo "Finished Experiment"
date
echo "######################"
