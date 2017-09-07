# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.client import device_lib

sys.path.append('../TensorLib')
from Quantize import QConv
from Quantize import QFullyConnect
from Quantize import QBatchNorm
from Quantize import Factories
from Quantize import Quantizers

slim = tf.contrib.slim


def quantizer_selector(selector_str, **kwargs):
    quantizer=None
    if selector_str=="zero":
        quantizer = Quantizers.FixedPointQuantizer_zero(
                                    kwargs['quant_width'], kwargs['quant_prec'])
    elif selector_str=="down":
        quantizer = Quantizers.FixedPointQuantizer_down(
                                    kwargs['quant_width'], kwargs['quant_prec'])
    elif selector_str=="nearest":
        quantizer = Quantizers.FixedPointQuantizer_nearest(
                                    kwargs['quant_width'], kwargs['quant_prec'])
    elif selector_str=="stochastic":
        quantizer = Quantizers.FixedPointQuantizer_stochastic(
                                    kwargs['quant_width'], kwargs['quant_prec'])
    else:
        raise ValueError('Quantizer %s not recognized!'%(selector_str))
    return quantizer        


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def quantizer_map(quantizer_str, quantize_layers):
    quant_width=0
    quant_prec=0
    rounding=''
    
    # make a list from the layers to be quantized
    if quantize_layers !="":
        q_layers = quantize_layers.split(",")
    else:
        q_layers = None

    # get the quantizer parameters
    tokens = quantizer_str.split(',')
    if len(tokens)>=2:
        quant_width = int(tokens[0])
        quant_prec = int(tokens[1])
    if len(tokens)>=3:
        rounding = tokens[2]

    # generate quantizer dictionary
    if quantizer_str !='' and q_layers is not None:
        if quant_width > quant_prec and quant_prec >= 0:
            q_map={}
            for key in q_layers:
                q_map[key]=quantizer_selector(rounding, quant_width=quant_width, quant_prec=quant_prec)
        else:
            raise ValueError('Quantizer initialized with invalid values: (%d,%d)'
                                %(quant_width,quant_prec))
    else:
        q_map=None

    return (q_map, quant_width, quant_prec, rounding)

