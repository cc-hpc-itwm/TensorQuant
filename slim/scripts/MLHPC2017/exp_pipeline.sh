#!/bin/bash
#
# Dominik Loroch
# 
# Experiment Pipeline

DATASET_DIR=/data/tf

echo "Pipeline PID:" $$


#./scripts/experiments/A_gpu1.sh ${DATASET_DIR} &
./scripts/experiments/A_gpu1.sh ${DATASET_DIR}

echo "Pipeline finished."
