#!/bin/bash
#
# Dominik Loroch
# 
# Experiment Pipeline

GPUs=2

for (( i=0; i<$GPUs; ++i ))
do
    cp A_gpu_template.sh B_gpu${i}.sh
    echo 'echo "GPU'${i}' PID:" $$' >>B_gpu${i}.sh
    echo 'export CUDA_VISIBLE_DEVICES='${i} >>B_gpu${i}.sh
done

gpu () {
    echo "B_gpu${1}.sh"
}

# rounding layerwise experiments
# ---------------------------
echo './scripts/experiments/template_rounding_layerwise.sh "inception_v1" "./tmp/inceptionv1-model" \
                            ${DATASET_DIR} 200 "${inceptionv1_layers}"' >>$(gpu 0)
echo './scripts/experiments/template_rounding_layerwise.sh "inception_v3" "./tmp/inceptionv3-model" \
                            ${DATASET_DIR} 200 "${inceptionv3_layers}"' >>$(gpu 0)
echo './scripts/experiments/template_rounding_layerwise.sh "resnet_v1_50" "./tmp/resnetv1_50-model" \
                            ${DATASET_DIR} 200 "${resnet50_layers}"' >>$(gpu 0)
echo './scripts/experiments/template_rounding_layerwise.sh "resnet_v1_152" "./tmp/resnetv1_152-model" \
                            ${DATASET_DIR} 200 "${resnet152_layers}"' >>$(gpu 3)

# layerwise experiments
# ---------------------------
echo './scripts/experiments/template_layerwise.sh "inception_v1" "./tmp/inceptionv1-model" \
                            ${DATASET_DIR} 200 "${inceptionv1_layers}"' >>$(gpu 0)
echo './scripts/experiments/template_layerwise.sh "inception_v3" "./tmp/inceptionv3-model" \
                            ${DATASET_DIR} 200 "${inceptionv3_layers}"' >>$(gpu 0)
echo './scripts/experiments/template_layerwise.sh "resnet_v1_50" "./tmp/resnetv1_50-model" \
                            ${DATASET_DIR} 200 "${resnet50_layers}"' >>$(gpu 0)
echo './scripts/experiments/template_layerwise.sh "resnet_v1_152" "./tmp/resnetv1_152-model" \
                            ${DATASET_DIR} 200 "${resnet152_layers}"' >>$(gpu 1)

# rounding all layers experiments
# ---------------------------
echo './scripts/experiments/template_rounding_all.sh "inception_v1" "./tmp/inceptionv1-model" \
                            ${DATASET_DIR} 200' >>$(gpu 4)
echo './scripts/experiments/template_rounding_all.sh "inception_v3" "./tmp/inceptionv3-model" \
                            ${DATASET_DIR} 200' >>$(gpu 5)
echo './scripts/experiments/template_rounding_all.sh "resnet_v1_50" "./tmp/resnetv1_50-model" \
                            ${DATASET_DIR} 200' >>$(gpu 6)
echo './scripts/experiments/template_rounding_all.sh "resnet_v1_152" "./tmp/resnetv1_152-model" \
                            ${DATASET_DIR} 200' >>$(gpu 7)

# all layers experiments
# ---------------------------
echo './scripts/experiments/template_all.sh "inception_v1" "./tmp/inceptionv1-model" \
                            ${DATASET_DIR} 200' >>$(gpu 7)
echo './scripts/experiments/template_all.sh "inception_v3" "./tmp/inceptionv3-model" \
                            ${DATASET_DIR} 200' >>$(gpu 6)
echo './scripts/experiments/template_all.sh "resnet_v1_50" "./tmp/resnetv1_50-model" \
                            ${DATASET_DIR} 200' >>$(gpu 5)
echo './scripts/experiments/template_all.sh "resnet_v1_152" "./tmp/resnetv1_152-model" \
                            ${DATASET_DIR} 200' >>$(gpu 4)


# batchnorm experiments
# ---------------------------
echo './scripts/experiments/template_batchnorm.sh "inception_v1" "./tmp/inceptionv1-model" \
                            ${DATASET_DIR} 200' >>$(gpu x)
echo './scripts/experiments/template_batchnorm.sh "inception_v3" "./tmp/inceptionv3-model" \
                            ${DATASET_DIR} 200' >>$(gpu x)
echo './scripts/experiments/template_batchnorm.sh "resnet_v1_50" "./tmp/resnetv1_50-model" \
                            ${DATASET_DIR} 200' >>$(gpu x)
echo './scripts/experiments/template_batchnorm.sh "resnet_v1_152" "./tmp/resnetv1_152-model" \
                            ${DATASET_DIR} 200' >>$(gpu x)


for (( i=0; i<$GPUs; ++i ))
do
    echo 'unset CUDA_VISIBLE_DEVICES' >>B_gpu${i}.sh
done


