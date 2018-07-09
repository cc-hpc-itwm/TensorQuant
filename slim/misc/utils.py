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
import re
import os
from functools import reduce
import operator

from tensorflow.python.client import device_lib

slim = tf.contrib.slim


def get_available_gpus():
    """ Returns available GPUs.
    Returns:
        List of available GPUs.
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_nb_params_shape(shape):
        """ Computes the total number of params for a given shape.
        Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
        Args:
            shape: A list of integers, representing the shape of the tensor.
        Return:
            The number of parameters in the shape.
        """
        return reduce(operator.mul, [int(s) for s in shape], 1)


def count_trainable_params(var_name):
    """ Counts the number of trainable parameters in a variable.
    Args:
        var_name: the name of the variable
    Return:
        number of parameters in the variable
    """
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        if var_name in trainable_variable.name:
            shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
            current_nb_params = get_nb_params_shape(shape)
            tf.logging.info('%s: %d'%(trainable_variable.name, current_nb_params))
            tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params


def get_variables_list(keyword):
    '''
    Returns a list with the names of the quantized variables and 
    a list with the tensors of the quantized variables. If there
    is no quantized version, the regular variable is used. Can 
    filter for 'weights' or 'biases'.
    Arg:
        keyword: either filter for 'weights' or 'biases'
    Returns:
        Two lists, one with strings and one with corresponding tensor objects.
    '''
    # check keyword
    if keyword != "weights" and keyword != "biases":
        raise ValueError("Expected keyword one of 'weights' or 'biases', got: %s"%keyword)
    
    # get varibles of desired type
    variable_list = tf.trainable_variables()
    key_list = [key.name.replace(":0","") for key in variable_list if keyword in key.name]

    # get a list of all tensor names
    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    
    key_name_list = [] # list of final names of variables

    # compare all variable names against tensor names and check if there are quantized versions of the variable available in the tensor list
    for var in key_list:
        quant_version_exists=0
        for tensor_name in tensor_name_list:
            # subtract the "weights"/"biases" from var name
            reduced_var=var.replace("weights","").replace("biases","")
            if reduced_var in tensor_name and ("quant_"+keyword) in tensor_name:
                quant_version_exists=1
                key_name_list.append(tensor_name)
                break
        if quant_version_exists is 0:
                key_name_list.append(var)

    # compile a list of tensors, which belong to the found variables
    key_tensor_list = [ tf.get_default_graph().get_tensor_by_name(name+':0')
                        for name in key_name_list ]
    return key_name_list, key_tensor_list


def get_vals_from_comments(key, val_re, data):
    """ Extracts values from comment string.
        The values are attached to the data elements.
        Args:
            key: A token in the comment string, followed by '='
            val_re: regular expression of the value, e.g.
                '([\w/]*)' for words
                '(\d*\.?\d*)' for numbers
                re must be in ()!
            data: a list, whose elements are dictionaries with a field 'comment'

        Return:
            A list of all occuring values for the given key.
    """
    _list = []
    for x in data:
        _regex = re.compile(key+'='+val_re)
        tokens = _regex.search(x['comment'])
        if tokens is None:
            x[key] = None
            continue
        val = tokens.group(1)
        try:
            val=float(val)
        except:
            pass
        x[key] = val
        if val not in _list:
            _list.append(val)
    #_list=sorted(_list)
    return _list


def remove_file(filename):
    """ Silently remove a file.
        Does nothing if there is no file to remove.
    """
    try:
        os.remove(filename)
    except OSError:
        pass


def get_all_variables_as_single_op(key):
    """ Get all variables of type 'weights' or 'biases' as a single list.
        Args:
            key: either "weights" or "biases"
        Return:
            One tensor with one dimension of the combined sizes of all variables.
        
    """
    weights_name_list, weights_list = get_variables_list(key)

    _op=[ tf.reshape(x,[tf.size(x)]) for x in weights_list]
    _op=tf.concat(_op,axis=0)
    return _op


def get_variables_count_dict(key):
    """ Returns a dictionary, which maps variable names to their respective count of parameters.
    Args:
        key: either "weights" or "biases"
    Return:
        A dictionary with name and count pairs.
    """

    # get the quantized weight tensors for sparsity estimation
    weights_name_list, weights_list = get_variables_list(key)

    # count number of elements in each layer
    weights_list_param_count = [ get_nb_params_shape(x.get_shape()) 
                                    for x in weights_list]
    # make a dictionary between variable names and number of parameters
    weights_list_param_count = dict(zip(weights_name_list, weights_list_param_count))
    return weights_list_param_count


def get_sparsity_ops(key):
    """ Returns sparsity ops for layerwise and total sparsity.
    Args:
        key: either "weights" or "biases"
    Return:
        A list of layerwise and total sparsity ops for variables.
    """
    # get the quantized weight tensors for sparsity estimation
    vars_name_list, vars_list = get_variables_list(key)

    # get zero fraction for each layer
    vars_layerwise_sparsity_op=[ tf.nn.zero_fraction(x) for x in vars_list]

    # add total varss sparsity to summary
    vars_total_sparsity_op=[ tf.reshape(x,[tf.size(x)]) for x in vars_list]
    vars_total_sparsity_op=tf.concat(vars_total_sparsity_op,axis=0)
    vars_total_sparsity_op=tf.nn.zero_fraction(vars_total_sparsity_op)
    return vars_layerwise_sparsity_op, vars_total_sparsity_op


def find_char_positions_in_string(char, string):
    """ Returns a list which contains the positions of the single character char
    within the string.
    """
    return [pos for pos, c in enumerate(string) if c == char]


def reduce_hierarchy_list(str_list, level):
    """ Reduces every string in str_list to a position defined by level:
    (example with level=2):

    InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1
    becomes
    InceptionV1/Mixed_3b

    Levels start at 1.
    The returned list does not contain double entries.
    """
    separator='/'

    reduced_list=[]
    for it in str_list:
        positions= find_char_positions_in_string(separator, it)
        if not positions:
            new_element= it
        if level>0:
            if len(positions) < level:
                new_element= it
            else:
                new_element= it[:positions[level-1]]
        else:
            if len(positions)< abs(level):
                new_element=None
            else:
                new_element=it[:positions[len(positions)-abs(level)]]
        if new_element not in reduced_list:
            reduced_list.append(new_element)
    return reduced_list


def remove_net_prefix(str_list):
    """ Removes the first word before "/" in every string in the list str_list.
    """
    separator='/'

    reduced_list=[]
    for it in str_list:
        positions= find_char_positions_in_string(separator, it)
        if not positions:
            new_element= it
        else:
            new_element= it[positions[0]+1:]
        if new_element not in reduced_list:
            reduced_list.append(new_element)
    return reduced_list


def get_variables_containing(string):
    """ Searches tf.trainable_variables for variables with string in their names.
    Returns a list of matching variables.
    """
    variable_list = []
    for var in tf.trainable_variables():
        if string in var.name:
            variable_list.append(var)
    return variable_list


def find_tensors_containing(string):
    """ Searches the default graph for tensors with string in their names.
    Returns a list of matching node descriptors.
    """
    tensor_list = []
    for tensor in tf.get_default_graph().as_graph_def().node:
        if string in tensor.name:
            tensor_list.append(tensor)
    return tensor_list


def get_tensor_containing(string):
    """ Gets the first tensor with string in its name.
    Returns a single tensor object, or None if there is no hit.
    """
    for tensor in tf.get_default_graph().as_graph_def().node:
        if string in tensor.name:
            return tf.get_default_graph().get_tensor_by_name(tensor.name+':0')
    return None


def get_tensor(string):
    """ Get the tensor whose name matches string.
    Returns a single tensor object, or None if there is no hit.
    """
    for tensor in tf.get_default_graph().as_graph_def().node:
        if string == tensor.name:
            return tf.get_default_graph().get_tensor_by_name(tensor.name+':0')
    return None


def find_leaf_nodes(tensor_name):
        """ Returns a list of all node names which have the given tensor as input
        Args:
            tensor_name: name of a tensor
        Return:
            A list of node names, which have 
        """
        leaves=[]
        for node in tf.get_default_graph().as_graph_def().node:
            for _input in node.input:
                if tensor_name in _input:
                    leaves.append(tensor.name)
        return leaves
