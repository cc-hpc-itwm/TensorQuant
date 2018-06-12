#!/bin/bash
#
# Start script from slim/ directory!

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=./tmp/lenet-model/baseline

# Where the dataset is saved to.
DATASET_DIR=~/tmp/mnist

# Download the dataset
#python download_and_convert_data.py \
#  --dataset_name=mnist \
#  --dataset_dir=${DATASET_DIR}

#export CUDA_VISIBLE_DEVICES=0
# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/tf \
  --dataset_name=mnist \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet \
  --max_num_batches=-1 \
  --batch_size=512 \
#  --weight_qmap=${TRAIN_DIR}/QMaps/weights.json \
#  --output_file=./experiment_results/test_$$.json \
#  --comment='Run Description.' \
#  --extr_qmap=${TRAIN_DIR}/QMaps/weights.json \
#  --intr_qmap=${TRAIN_DIR}/QMaps/weights.json
