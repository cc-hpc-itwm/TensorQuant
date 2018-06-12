#!/bin/bash
#
# Dominik Loroch
# 
# Experiment Pipeline

DATASET_DIR=${1}


echo "GPU1 PID:" $$

export CUDA_VISIBLE_DEVICES=1

#./scripts/train_experiments/alexnet_train_sparse_extr.sh "0.001"
./scripts/train_experiments/alexnet_train_sparse_extr.sh "0.0001"

unset CUDA_VISIBLE_DEVICES
