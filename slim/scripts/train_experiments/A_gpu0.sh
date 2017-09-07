#!/bin/bash
#
# Dominik Loroch
# 
# Experiment Pipeline

DATASET_DIR=${1}


echo "GPU0 PID:" $$

export CUDA_VISIBLE_DEVICES=0

./scripts/train_experiments/lenet_train.sh "lenet" "./tmp/lenet-model" \
                         ${DATASET_DIR} -1

unset CUDA_VISIBLE_DEVICES
