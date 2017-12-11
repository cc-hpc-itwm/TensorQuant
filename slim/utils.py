# Utilities used in the evaluation and training python scripts.
#
# author: Dominik Loroch
# date: August 2017

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import json

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.client import device_lib

from Quantize import Quantizers

EPS=0.000001
slim = tf.contrib.slim


def quantizer_selector(selector_str, arg_list):
    """ Builds and returns the specified quantizer.
    Args:
        selector_str: The name of the quantizer.
        arg_list: Arguments which need to be passed to the constructor of the quantizer.
    Returns:
        Quantizer object.
    """
    if selector_str=="zero":
        quantizer = Quantizers.FixedPointQuantizer_zero(
                                    int(arg_list[0]), int(arg_list[1]) )
    elif selector_str=="down":
        quantizer = Quantizers.FixedPointQuantizer_down(
                                    int(arg_list[0]), int(arg_list[1]) )
    elif selector_str=="nearest":
        quantizer = Quantizers.FixedPointQuantizer_nearest(
                                    int(arg_list[0]), int(arg_list[1]) )
    elif selector_str=="stochastic":
        quantizer = Quantizers.FixedPointQuantizer_stochastic(
                                    int(arg_list[0]), int(arg_list[1]) )
    elif selector_str=="sparse":
        quantizer = Quantizers.SparseQuantizer(
                                    float(arg_list[0]) )
    elif selector_str=="logarithmic":
        quantizer = Quantizers.LogarithmicQuantizer(
                                    )
    else:
        raise ValueError('Quantizer %s not recognized!'%(selector_str))
    return quantizer        


def get_available_gpus():
    """ Returns available GPUs.
    Returns:
        List of available GPUs.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def split_quantizer_str(quantizer_str):
    """ Splits a comma seperated list into its components.
        Interprets the first entry as the quantizer name.
    Args:
        quantizer_str: String in the form: "quantizer_type,argument_1,argument_2,..."
    Returns:
        Tupel of strings in the form (quantizer_type, [argument_1, argument_2,...])
    """
    quantizer_type=''
    args=[]
    tokens = quantizer_str.split(',')
    if len(tokens) > 0:
        quantizer_type=tokens[0]
    if len(tokens) > 1:
        args=tokens[1:]
    return (quantizer_type, args)


def split_layers_str(layers_str):
    """ Splits a comma seperated list into its components.
    Strips leading and trailing blanks from each entry.
    Args:
        layers_str: String in the form: "layer_1,layer_2,..."
    Returns:
        List of strings in the form [layer_1, layer_2, ...]
    """
    if layers_str !="":
        layers = layers_str.split(",")
        layers = [layer.strip() for layer in layers]
    else:
        layers = []
    return layers

def quantizer_map(qmap_file):
    """ Creates a Quantizer map. All specified layers share the same quantizer type.
    Args:
        qmap_file: Location of the .json file, which specifies the mapping.
    Returns:
        A dictionary containing the mapping from layers to quantizers.
    """
    # load dictionary from json file.
    # open file and parse data
    if qmap_file is '':
        return None
    with open(qmap_file,'r') as hfile:
        qmap = json.load(hfile)

    # change strings in qmap into quantizer objects
    for key in qmap:
      if type(qmap[key]) is str:
        # get the quantizer parameters
        quantizer_type, arg_list = split_quantizer_str(qmap[key])
        # generate quantizer object
        qmap[key]=quantizer_selector(quantizer_type, arg_list)
    return qmap

def count_trainable_params(var_name):
    '''
    Counts the number of trainable variables.
    '''
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        if var_name in trainable_variable.name:
            shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
            current_nb_params = get_nb_params_shape(shape)
            tf.logging.info('%s: %d'%(trainable_variable.name, current_nb_params))
            tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params

def get_nb_params_shape(shape):
    '''
    Computes the total number of params for a given shap.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    '''
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params 

def compute_sparsity(data):
    data=np.array(data)
    data=data.flatten()
    count=sum(abs(data)<=EPS)
    sparsity=count/len(data)
    return sparsity

def get_variables_name_list(keyword, var_list):
    _name_list = []
    tensor_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    for var in var_list:
        quant_version_exists=0
        for tensor in tensor_list:
            # subtract the "weights"/"biases" from var name
            if var[:-len(keyword)] in tensor and ("quant_"+keyword) in tensor:
                quant_version_exists=1
                if _name_list is []:
                    _name_list = [tensor]
                else:
                    _name_list.append(tensor)
        if quant_version_exists is 0:
            if _name_list is []:
                _name_list = [var]
            else:
                _name_list.append(var)
    return _name_list


