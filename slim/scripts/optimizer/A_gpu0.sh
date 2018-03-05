#!/bin/bash


DATASET_DIR=${1}

echo "GPU0 PID:" $$

export CUDA_VISIBLE_DEVICES=0

./scripts/optimizer/sparse/sparse_opt_lenet.sh
./scripts/optimizer/sparse/sparse_opt_alexnet.sh
./scripts/optimizer/sparse/sparse_opt_inceptionv1.sh
./scripts/optimizer/sparse/sparse_opt_inceptionv3.sh
./scripts/optimizer/sparse/sparse_opt_resnet50.sh
./scripts/optimizer/sparse/sparse_opt_resnet152.sh

unset CUDA_VISIBLE_DEVICES
