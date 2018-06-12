#!/bin/bash
#
# Dominik Loroch
# 
# Experiment Pipeline

DATASET_DIR=${1}

echo "GPU1 PID:" $$

export CUDA_VISIBLE_DEVICES=1

./scripts/experiments/sparsity/sparse_layerwise.sh "inception_v1" "./tmp/inceptionv1-model" \
                            ${DATASET_DIR} 200
./scripts/experiments/sparsity/sparse_layerwise.sh "inception_v3" "./tmp/inceptionv3-model" \
                            ${DATASET_DIR} 200
./scripts/experiments/sparsity/sparse_layerwise.sh "alexnet_v2" "./tmp/alexnet-model" \
                            ${DATASET_DIR} 200
./scripts/experiments/sparsity/sparse_layerwise.sh "resnet_v1_50" "./tmp/resnetv1_50-model" \
                            ${DATASET_DIR} 200
./scripts/experiments/sparsity/sparse_layerwise.sh "resnet_v1_152" "./tmp/resnetv1_152-model" \
                            ${DATASET_DIR} 200
./scripts/experiments/sparsity/sparse_layerwise.sh "lenet" "./tmp/lenet-model" \
                            ${DATASET_DIR} 200

unset CUDA_VISIBLE_DEVICES

