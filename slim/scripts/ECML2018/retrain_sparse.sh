#!/bin/bash

echo "GPU0 PID:" $$

#export CUDA_VISIBLE_DEVICES=0

# weight sparsity
./scripts/ECML2018/retrain/lenet.sh "./tmp/lenet-model" "lenet" "mnist"
./scripts/ECML2018/retrain/alexnet.sh "./tmp/alexnet-model" "alexnet_v2" "imagenet"
#./scripts/ECML2018/sparse/sparse_opt_alexnet.sh "weight"
#./scripts/ECML2018/sparse/sparse_opt_inceptionv1.sh "weight"
#./scripts/ECML2018/sparse/sparse_opt_inceptionv3.sh "weight"
#./scripts/ECML2018/sparse/sparse_opt_resnet50.sh "weight"
#./scripts/ECML2018/sparse/sparse_opt_resnet152.sh "weight"

#unset CUDA_VISIBLE_DEVICES
