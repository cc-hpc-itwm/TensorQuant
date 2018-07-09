# Utilities for Quantizers used in the evaluation and training python scripts.
#
# author: Dominik Loroch
# date: August 2017

import json
import tensorflow as tf

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
    if selector_str=="none":
        quantizer = Quantizers.NoQuantizer()
    elif selector_str=="zero":
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
        quantizer = Quantizers.LogarithmicQuantizer()
    elif selector_str=="fp16":
        quantizer = Quantizers.HalffpQuantizer()
    else:
        raise ValueError('Quantizer %s not recognized!'%(selector_str))
    return quantizer


def split_quantizer_str(quantizer_str):
    """ Splits a quantizer string into its components.
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
