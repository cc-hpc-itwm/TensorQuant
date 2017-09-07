#!/bin/bash
#
# Dominik Loroch
# 
# Experiment Pipeline

DATASET_DIR=/mnt/beegfs/tf

echo "Pipeline PID:" $$


./scripts/experiments/B_gpu1.sh ${DATASET_DIR} &
./scripts/experiments/A_gpu0.sh ${DATASET_DIR}

echo "Pipeline finished."
