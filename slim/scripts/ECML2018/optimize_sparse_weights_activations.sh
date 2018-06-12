#!/bin/bash

echo "GPU0 PID:" $$

export CUDA_VISIBLE_DEVICES=0

# activation sparsity
./scripts/ECML2018/sparse/sparse_opt_lenet.sh "extr"
./scripts/ECML2018/sparse/sparse_opt_alexnet.sh "extr"
./scripts/ECML2018/sparse/sparse_opt_inceptionv1.sh "extr"
./scripts/ECML2018/sparse/sparse_opt_inceptionv3.sh "extr"
./scripts/ECML2018/sparse/sparse_opt_resnet50.sh "extr"
./scripts/ECML2018/sparse/sparse_opt_resnet152.sh "extr"

# weight sparsity
./scripts/ECML2018/sparse/sparse_opt_lenet.sh "weight"
./scripts/ECML2018/sparse/sparse_opt_alexnet.sh "weight"
./scripts/ECML2018/sparse/sparse_opt_inceptionv1.sh "weight"
./scripts/ECML2018/sparse/sparse_opt_inceptionv3.sh "weight"
./scripts/ECML2018/sparse/sparse_opt_resnet50.sh "weight"
./scripts/ECML2018/sparse/sparse_opt_resnet152.sh "weight"

unset CUDA_VISIBLE_DEVICES
