#!/bin/bash

echo "GPU0 PID:" $$

export CUDA_VISIBLE_DEVICES=0

# activation sparsity
./scripts/ECML2018/eval/lenet.sh 
./scripts/ECML2018/eval/alexnet.sh
./scripts/ECML2018/eval/inceptionv1.sh
./scripts/ECML2018/eval/inceptionv3.sh
./scripts/ECML2018/eval/resnet50.sh
./scripts/ECML2018/eval/resnet152.sh

unset CUDA_VISIBLE_DEVICES
