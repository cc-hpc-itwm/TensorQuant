# Utilities for Quantizers used in the evaluation and training python scripts.
#
# author: Dominik Loroch
# date: August 2017

import json
import tensorflow as tf

from TensorQuant.Quantize import Quantizers

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
    elif selector_str=="binary":
        if len(arg_list)==0:
            quantizer = Quantizers.BinaryQuantizer( 1 )
        if len(arg_list)==1:
            quantizer = Quantizers.BinaryQuantizer( float(arg_list[0]) )
    elif selector_str=="ternary":
        if len(arg_list)==0:
            quantizer = Quantizers.TernaryQuantizer( 1 )
        if len(arg_list)==1:
            quantizer = Quantizers.TernaryQuantizer( float(arg_list[0]) )
        elif len(arg_list)==2:
            quantizer = Quantizers.TernaryQuantizer( float(arg_list[0]), False, float(arg_list[1]))
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


def get_quantizer(q_str):
    """ Get a quantizer instance based on string.
        If quantizer is empty string or None, None is returned.
    Args:
        q_str: quantizer string to be interpreted.
    Returns:
        Quantizer object or None.
    """
    if q_str == "":
        q_str=None
    if q_str is None:
        return None
    qtype, qargs= split_quantizer_str(q_str)
    quantizer = quantizer_selector(qtype, qargs)
    return quantizer


def quantizer_map(qmap):
    """ Creates a Quantizer map. All specified layers share the same quantizer type.
    Args:
        qmap: Location of the .json file, which specifies the mapping, or a dictionary with the same content.
    Returns:
        A dictionary containing the mapping from layers to quantizers.
    """
    if qmap is None:
        return None
    elif type(qmap) == str:
        # load dictionary from json file.
        # open file and parse data
        if qmap is '':
            return None
        try:
            with open(qmap,'r') as hfile:
                qmap = json.load(hfile)
        except IOError:
            qmap={"":qmap}

    # change strings in qmap into quantizer objects
    for key in qmap:
      if type(qmap[key]) is str:
        # generate quantizer object
        quantizer=get_quantizer(qmap[key])
        #if quantizer is None:
        #    raise ValueError("Invalid quantizer \""+qmap[key]+"\" for layer \""+key+"\"")
        qmap[key]=quantizer

    return qmap
