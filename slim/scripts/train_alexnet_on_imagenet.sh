#!/bin/bash
#
# Start script from slim/ directory!

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/alexnet-model/base_rmsprop_02
#rm -r ${TRAIN_DIR}
mkdir ${TRAIN_DIR}

# Name of Dataset
DATASET_NAME=imagenet

# Where the dataset is saved to.
DATASET_DIR=/data/tf

echo "######################"
echo "Start training"
date
echo "######################"
# Run training.
#export CUDA_VISIBLE_DEVICES=0

#Reference: 450,000 steps with 128 batch, lr=0.1

python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=alexnet_v2 \
  --preprocessing_name=alexnet_v2 \
  --max_number_of_steps=450000 \
  --batch_size=128 \
  --num_clones=1 \
  --num_preprocessing_threads=16 \
  --save_interval_secs=3600 \
  --save_summaries_secs=3600 \
  --log_every_n_steps=1000 \
  --optimizer=rmsprop \
  --learning_rate=0.1 \
  --weight_decay=0.00004
  #--intr_grad_quantizer=logarithmic \
  #--weight_qmap=./tmp/alexnet-model/QMaps/weights.json
  #--intr_grad_quantizer=nearest,32,16

echo "######################"
echo "End training"
date
echo "######################"

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=alexnet_v2 \
