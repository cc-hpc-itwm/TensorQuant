#!/bin/bash
#
# Dominik Loroch
# 
# Experiment Pipeline

DATASET_DIR=${1}

resnet50_layers="resnet_v1_50/conv1
block1
block2
block3
block4
logits
"

resnet152_layers="resnet_v1_152/conv1
block1
block2
block3
block4
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

echo "GPUx PID:" $$
export CUDA_VISIBLE_DEVICES=x


# rounding layerwise experiments
# ---------------------------
#./scripts/experiments/template_rounding_layerwise.sh "inception_v1" "./tmp/inceptionv1-model" \
#                            ${DATASET_DIR} 200 ${inceptionv1_layers}
#./scripts/experiments/template_rounding_layerwise.sh "inception_v3" "./tmp/inceptionv3-model" \
#                            ${DATASET_DIR} 200 ${inceptionv3_layers}
#./scripts/experiments/template_rounding_layerwise.sh "resnet_v1_50" "./tmp/resnetv1_50-model" \
#                            ${DATASET_DIR} 200 ${resnet50_layers}
#./scripts/experiments/template_rounding_layerwise.sh "resnet_v1_152" "./tmp/resnetv1_152-model" \
#                            ${DATASET_DIR} 200 ${resnet152_layers}


# layerwise experiments
# ---------------------------
#./scripts/experiments/template_layerwise.sh "inception_v1" "./tmp/inceptionv1-model" \
#                            ${DATASET_DIR} 200 ${inceptionv1_layers}
#./scripts/experiments/template_layerwise.sh "inception_v3" "./tmp/inceptionv3-model" \
#                            ${DATASET_DIR} 200 ${inceptionv3_layers}
#./scripts/experiments/template_layerwise.sh "resnet_v1_50" "./tmp/resnetv1_50-model" \
#                            ${DATASET_DIR} 200 ${resnet50_layers}
#./scripts/experiments/template_layerwise.sh "resnet_v1_152" "./tmp/resnetv1_152-model" \
#                            ${DATASET_DIR} 200 ${resnet152_layers}

# rounding all layers experiments
# ---------------------------
#./scripts/experiments/template_rounding_all.sh "inception_v1" "./tmp/inceptionv1-model" \
#                            ${DATASET_DIR} 200
#./scripts/experiments/template_rounding_all.sh "inception_v3" "./tmp/inceptionv3-model" \
#                            ${DATASET_DIR} 200
#./scripts/experiments/template_rounding_all.sh "resnet_v1_50" "./tmp/resnetv1_50-model" \
#                            ${DATASET_DIR} 200
#./scripts/experiments/template_rounding_all.sh "resnet_v1_152" "./tmp/resnetv1_152-model" \
#                            ${DATASET_DIR} 200

# all layers experiments
# ---------------------------
#./scripts/experiments/template_all.sh "inception_v1" "./tmp/inceptionv1-model" \
#                            ${DATASET_DIR} 200
#./scripts/experiments/template_all.sh "inception_v3" "./tmp/inceptionv3-model" \
#                            ${DATASET_DIR} 200
#./scripts/experiments/template_all.sh "resnet_v1_50" "./tmp/resnetv1_50-model" \
#                            ${DATASET_DIR} 200
#./scripts/experiments/template_all.sh "resnet_v1_152" "./tmp/resnetv1_152-model" \
#                            ${DATASET_DIR} 200


# batchnorm experiments
# ---------------------------
#./scripts/experiments/template_batchnorm.sh "inception_v1" "./tmp/inceptionv1-model" \
#                            ${DATASET_DIR} 200
#./scripts/experiments/template_batchnorm.sh "inception_v3" "./tmp/inceptionv3-model" \
#                            ${DATASET_DIR} 200
#./scripts/experiments/template_batchnorm.sh "resnet_v1_50" "./tmp/resnetv1_50-model" \
#                            ${DATASET_DIR} 200
#./scripts/experiments/template_batchnorm.sh "resnet_v1_152" "./tmp/resnetv1_152-model" \
#                            ${DATASET_DIR} 200




unset CUDA_VISIBLE_DEVICES

