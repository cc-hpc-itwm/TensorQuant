#!/bin/bash
#
# Dominik Loroch
# 
# Experiment Pipeline

DATASET_DIR=${1}


echo "GPU0 PID:" $$

export CUDA_VISIBLE_DEVICES=0

#./scripts/train_experiments/alexnet_train_sparse_extr.sh "0.1"
./scripts/train_experiments/alexnet_train_sparse_extr.sh "0.01"



unset CUDA_VISIBLE_DEVICES
