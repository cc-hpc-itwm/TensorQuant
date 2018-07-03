#!/bin/bash
#

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/cifarnet-model/baseline
#TRAIN_DIR=./tmp/cifarnet-model/l1-l2-regularizer

# Where the dataset is saved to.
DATASET_DIR=~/tmp/cifar10

# Run training.
#export CUDA_VISIBLE_DEVICES=0

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=cifar10 \
  --dataset_split_name=test \
  --max_num_batches=-1 \
  --batch_size=128 \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet \
  --comment='baseline' \
  --output_file='./experiment_results/cifarnet_baseline.json' \
  #--weight_qmap=${TRAIN_DIR}/opt_weight.json
