#!/bin/bash

THRESHOLDS="0.01
0.001
0.0001
0.00008
0.00006
0.00005
0.00004
0.00003
0.00001
0.000001"

./scripts/ECML2018/overfitting/resnet50_overfitting.sh "${THRESHOLDS}"
#./scripts/ECML2018/overfitting/resnet20_overfitting.sh "${THRESHOLDS}"
#./scripts/ECML2018/overfitting/resnet14_overfitting.sh "${THRESHOLDS}"
