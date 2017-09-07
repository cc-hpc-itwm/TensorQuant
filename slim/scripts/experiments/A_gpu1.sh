#!/bin/bash
#
# Dominik Loroch
# 
# Experiment Pipeline

DATASET_DIR=${1}

echo "GPU1 PID:" $$

export CUDA_VISIBLE_DEVICES=1

./scripts/experiments/lenet_all.sh

unset CUDA_VISIBLE_DEVICES

