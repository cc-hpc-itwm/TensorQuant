#!/bin/bash
#
# Dominik Loroch
# 
# Experiment Pipeline

DATASET_DIR=~/tmp/mnist

echo "Pipeline PID:" $$


./scripts/train_experiments/A_gpu0.sh ${DATASET_DIR}


echo "Pipeline finished."
