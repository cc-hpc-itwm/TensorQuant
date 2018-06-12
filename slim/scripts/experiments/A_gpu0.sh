#!/bin/bash
#
# Dominik Loroch
# 
# Experiment Pipeline

DATASET_DIR=${1}

echo "GPU0 PID:" $$

export CUDA_VISIBLE_DEVICES=0

./scripts/experiments/sparse_extWeight.sh "alexnet_v2" "./tmp/alexnet-model" \
                            ${DATASET_DIR} 200

#./scripts/experiments/googlenet_sparsity.sh

unset CUDA_VISIBLE_DEVICES
