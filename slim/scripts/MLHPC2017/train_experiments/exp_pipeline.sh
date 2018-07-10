#!/bin/bash
#
# Dominik Loroch
# 
# Experiment Pipeline

DATASET_DIR=/data/tf

echo "Pipeline PID:" $$

./scripts/train_experiments/A_gpu0.sh ${DATASET_DIR} &
./scripts/train_experiments/A_gpu1.sh ${DATASET_DIR}


echo "Pipeline finished."
