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

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.client import device_lib

'''
from Quantize import QConv
from Quantize import QFullyConnect
from Quantize import QBatchNorm
from Quantize import Factories
'''
from Quantize import Quantizers

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


def quantizer_map(quantizer_str, quantize_layers):
    """ Creates a Quantizer map. All specified layers share the same quantizer type.
    Args:
        quantizer_str: A string specifying the quantizer.
        quantize_layers: Comma seperated string or list of strings, specifying the layers.
    Returns:
        A dictionary containing the mapping from layers to quantizers.
    """
    # make a list from the layers to be quantized
    if type(quantize_layers) is str:
        q_layers=split_layers_str(quantize_layers)

    # get the quantizer parameters
    quantizer_type, arg_list = split_quantizer_str(quantizer_str)

    # generate quantizer dictionary
    if quantizer_type !='' and len(q_layers)!=0:
        q_map={}
        for key in q_layers:
                q_map[key]=quantizer_selector(quantizer_type, arg_list)
    else:
        q_map=None
    return q_map

