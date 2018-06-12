#!/bin/bash
#
# 
# Experiment Pipeline

DATASET_DIR=/data/tf

echo "Pipeline PID:" $$


#./scripts/experiments/A_gpu1.sh ${DATASET_DIR} &
./scripts/optimizer/A_gpu0.sh ${DATASET_DIR}

echo "Pipeline finished."
