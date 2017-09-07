#!/bin/bash
#
# Dominik Loroch
# 
# Experiment Pipeline

DATASET_DIR=/mnt/beegfs/tf

resnet50_layers="resnet_v1_50/conv1
block1
block2
block3
block4
logits
"

resnet152_layers="block4
logits
"

inceptionv1_layers="Conv2d_1a_7x7
MaxPool_2a_3x3
Conv2d_2b_1x1
Conv2d_2c_3x3
MaxPool_3a_3x3
Mixed_3b
Mixed_3c
MaxPool_4a_3x3
Mixed_4b
Mixed_4c
Mixed_4d
Mixed_4e
Mixed_4f
MaxPool_5a_2x2
Mixed_5b
Mixed_5c
Logits
"

inceptionv3_layers="Conv2d_1a_3x3
Conv2d_2a_3x3
Conv2d_2b_3x3
MaxPool_3a_3x3
Conv2d_3b_1x1
Conv2d_4a_3x3
MaxPool_5a_3x3
Mixed_5b
Mixed_5c
Mixed_5d
Mixed_6a
Mixed_6b
Mixed_6c
Mixed_6d
Mixed_6e
Mixed_7a
Mixed_7b
Mixed_7c
AuxLogits
PreLogits
Logits
"



echo "GPU0 PID:" $$

export CUDA_VISIBLE_DEVICES=0

./scripts/experiments/template_layerwise.sh "resnet_v1_152" "./tmp/resnetv1_152-model" \
                            ${DATASET_DIR} 200 "${resnet152_layers}"

#./scripts/experiments/template_extr_all.sh "inception_v1" "./tmp/inceptionv1-model" \
#                            ${DATASET_DIR} 200
#./scripts/experiments/template_extr_all.sh "inception_v3" "./tmp/inceptionv3-model" \
#                            ${DATASET_DIR} 200
#./scripts/experiments/template_extr_all.sh "resnet_v1_50" "./tmp/resnetv1_50-model" \
#                            ${DATASET_DIR} 200
#./scripts/experiments/template_extr_all.sh "resnet_v1_152" "./tmp/resnetv1_152-model" \
#                            ${DATASET_DIR} 200

unset CUDA_VISIBLE_DEVICES
