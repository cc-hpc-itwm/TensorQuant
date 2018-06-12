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
import re
import os

#from datasets import dataset_factory
#from nets import nets_factory
#from preprocessing import preprocessing_factory
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

def get_variables_list(keyword):
    '''
    Returns a list with the names of the quantized variables and 
    a list with the tensors of the quantized variables. If there
    is no quantized version, the regular variable is used.
    Arg:
        keyword: either 'weights' or 'biases'
    Returns:
        Two lists as described above.
    '''
    variable_list = tf.trainable_variables()
    tensor_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    # subtract ":0" from name
    key_list = [key.name[:-2] for key in variable_list if keyword in key.name]
    key_name_list = []
    for var in key_list:
        quant_version_exists=0
        for tensor in tensor_list:
            # subtract the "weights"/"biases" from var name
            if var[:-len(keyword)] in tensor and ("quant_"+keyword) in tensor:
                quant_version_exists=1
                if key_name_list is []:
                    key_name_list = [tensor]
                else:
                    key_name_list.append(tensor)
        if quant_version_exists is 0:
            if key_name_list is []:
                key_name_list = [var]
            else:
                key_name_list.append(var)
    key_tensor_list = [ tf.get_default_graph().get_tensor_by_name(name+':0')
                        for name in key_name_list ]
    return key_name_list, key_tensor_list

def heatmap_conv(kernel, pad = 1):

    '''Visualize conv features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    see: gist.github.com/kukuruza/03731dc494603ceab0c5

    Args:
    kernel: tensor of shape [height, width, NumChannels, NumKernels]     
    pad: number of black pixels around each filter (between them)

    Return:
    Tensor of shape [batch=NumChannels,
                (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels=3].
    '''

    def factorization(n):
        for i in range(int(math.sqrt(float(n))), 0, -1):
            if n % i == 0:
                return (i, int(n / i))

    (grid_Y, grid_X) = factorization (kernel.get_shape().dims[3].value)
    # scale weights to [-1,1]
    #x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(tf.abs(kernel))
    kernel = (kernel) / (x_max)
    kernel_height = kernel.get_shape().dims[0].value
    kernel_width = kernel.get_shape().dims[1].value
    kernel_channels = kernel.get_shape().dims[2].value
    kernel_features = kernel.get_shape().dims[3].value
    #kernel shape: [height, width, channels, features]

    red_channel = (tf.ones_like(kernel)-tf.to_float(kernel>0)*kernel)
    green_channel = (tf.ones_like(kernel)-tf.abs(kernel))
    blue_channel = (tf.ones_like(kernel)+tf.to_float(kernel<0)*kernel)
    kernel = tf.transpose(tf.stack([red_channel,green_channel,blue_channel]), (4,1,2,3,0))
    #kernel shape: [features, height, width, channels, rgb]
    # channels will be the batch, rgb are the color channels. 
    # features will be reduced to single image

    # pad X and Y
    kernel = tf.pad(kernel, tf.constant( 
    [[0,0],[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')
    tile_height= kernel.get_shape().dims[1].value
    tile_width= kernel.get_shape().dims[2].value
    # 2*pad added to height and width

    # organize grid on Y axis
    kernel = tf.reshape(kernel, tf.stack([grid_X, tile_height * grid_Y, 
                                        tile_width, kernel_channels, 3]))
    # switch X and Y axes
    kernel = tf.transpose(kernel, (0, 2, 1, 3, 4))
    #kernel shape: [features, width, height, channels, rgb]
    # organize grid on X axis, drop 5th dimension
    kernel = tf.reshape(kernel, tf.stack([tile_width * grid_X, tile_height * grid_Y, 
                                            kernel_channels, 3]))
    kernel = tf.transpose(kernel, (2, 1, 0, 3))
    #kernel shape: [channels, height, width, rgb]

    '''
    # resize image, for better visibility
    old_shape = kernel.get_shape().as_list()
    new_shape = tf.stack([4*old_shape[1], 4*old_shape[2]])
    kernel = tf.image.resize_nearest_neighbor(kernel, new_shape, align_corners=True)
    '''
    return kernel


def heatmap_fullyconnect(kernel, pad = 1):
    '''Visualize fc-layer as an image.

    Args:
        kernel: tensor of shape [kernel_inputs, kernel_outputs]   
        pad: number of black pixels around filter

    Return:
    Tensor of shape [batch=1,
                kernel_inputs, kernel_outputs, NumChannels=3].
    '''

    # scale weights to [-1,1]
    x_max = tf.reduce_max(tf.abs(kernel))
    kernel = (kernel) / (x_max)
    kernel_height = kernel.get_shape().dims[0].value
    kernel_width = kernel.get_shape().dims[1].value
    #kernel shape: [height, width]

    red_channel = (tf.ones_like(kernel)-tf.to_float(kernel>0)*kernel)
    green_channel = (tf.ones_like(kernel)-tf.abs(kernel))
    blue_channel = (tf.ones_like(kernel)+tf.to_float(kernel<0)*kernel)
    kernel = tf.transpose(tf.stack([red_channel,green_channel,blue_channel]), (1,2,0))
    # kernel shape: [height, width, rgb]
    # add dimension for batch
    kernel = tf.reshape(kernel, tf.stack([1,kernel_height, kernel_width,3]))
    # kernel shape: [batch, height, width, rgb]

    # pad X and Y
    kernel = tf.pad(kernel, tf.constant( 
    [[0,0],[pad,pad],[pad, pad],[0,0]]), mode = 'CONSTANT')
    
    '''
    # resize image, for better visibility
    old_shape = kernel.get_shape().as_list()
    new_shape = tf.stack([4*old_shape[1], 4*old_shape[2]])
    kernel = tf.image.resize_nearest_neighbor(kernel, new_shape, align_corners=True)
    '''
    return kernel

def get_vals_from_comments(key, val_re, data):
    '''
        val_re must be a string in paranthesis!
    '''
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
    try:
        os.remove(filename)
    except OSError:
        pass

def get_all_variable_as_list(key):
    '''
    key is "weights" or "biases"
    '''
    weights_name_list, weights_list = get_variables_list(key)

    _op=[ tf.reshape(x,[tf.size(x)]) for x in weights_list]
    _op=tf.concat(_op,axis=0)
    return _op

def get_variables_count_dict(key):
    '''
    key is "weights" or "biases"
    '''

    # get the quantized weight tensors for sparsity estimation
    weights_name_list, weights_list = get_variables_list(key)

    # count number of elements in each layer
    weights_list_param_count = [ get_nb_params_shape(x.get_shape()) 
                                    for x in weights_list]
    weights_list_param_count = dict(zip(weights_name_list, weights_list_param_count))
    weights_overall_sparsity_op=[ tf.reshape(x,[tf.size(x)]) for x in weights_list]
    return weights_list_param_count

def get_sparsity_ops(key):
    '''
    key is "weights" or "biases"
    '''
    # get the quantized weight tensors for sparsity estimation
    weights_name_list, weights_list = get_variables_list(key)

    # get zero fraction for each layer
    weights_layerwise_sparsity_op=[ tf.nn.zero_fraction(x) for x in weights_list]

    # add overall weights sparsity to summary
    weights_overall_sparsity_op=[ tf.reshape(x,[tf.size(x)]) for x in weights_list]
    weights_overall_sparsity_op=tf.concat(weights_overall_sparsity_op,axis=0)
    summary_name = 'eval/%s_overall_sparsity'%key
    weights_overall_sparsity_op=tf.nn.zero_fraction(weights_overall_sparsity_op)
    op = tf.summary.scalar(summary_name, weights_overall_sparsity_op, collections=[])
    #op = tf.Print(op, [value], summary_name)
    tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
    return weights_layerwise_sparsity_op, weights_overall_sparsity_op


def find_char_positions_in_string(char, string):
    '''
    Returns a list which contains the positions of the single character char
    within the string.
    '''
    return [pos for pos, c in enumerate(string) if c == char]


def reduce_hierarchy_list(str_list, level):
    '''
    Reduces the list of strings str_list to a position defined by level, so that 
    (example with level=2):

    InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1
    becomes
    InceptionV1/Mixed_3b

    Levels start at 1.
    The returned list does not contain double entries.
    '''
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
    '''
    Removes the first word before "/"
    '''
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
    variable_list = []
    for var in tf.trainable_variables():
        if string in var.name:
            variable_list.append(var)
    return variable_list

def find_tensors_containing(string):
    tensor_list = []
    for tensor in tf.get_default_graph().as_graph_def().node:
        if string in tensor.name:
            tensor_list.append(tensor)
    return tensor_list

def get_tensor_containing(string):
    for tensor in tf.get_default_graph().as_graph_def().node:
        if string in tensor.name:
            return tf.get_default_graph().get_tensor_by_name(tensor.name+':0')
    return None

def get_tensor(string):
    for tensor in tf.get_default_graph().as_graph_def().node:
        if string == tensor.name:
            return tf.get_default_graph().get_tensor_by_name(tensor.name+':0')
    return None


def register_kfac(logits):
    layer_collection = tf.contrib.kfac.layer_collection.LayerCollection()
    weights_postfixes=['weights', 'kernels']
    biases_postfixes=['biases', 'bias']
    activation_postfixes=['BiasAdd','Conv2D','MatMul']

    #variable_list = tf.trainable_variables()
    #tensor_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    #print(variable_list)
    #print(tensor_list)
    conv_layers = reduce_hierarchy_list([t.name for t in find_tensors_containing('Conv2D')],-1)
    fc_layers = reduce_hierarchy_list([t.name for t in find_tensors_containing('MatMul')],-1)
    conv_weights = [get_variables_containing(t) for t in conv_layers]
    fc_weights = [get_variables_containing(t) for t in fc_layers]

    kfac_layers=[]
    for layer in conv_layers:
        layer_info={}
        layer_info['name']=layer
        layer_info['type']='Conv2D'
        kfac_layers.append(layer_info)
    for layer in fc_layers:
        layer_info={}
        layer_info['name']=layer
        layer_info['type']='MatMul'
        kfac_layers.append(layer_info)  

    #print(conv_layers)
    #print(conv_weights)
    #print(fc_layers)

    for item in kfac_layers:
        tf.logging.info('K-FAC Item: %s'%item)
        input=None
        output=None
        weights=None
        biases=None
        padding=None
        strides=None
        if 'Conv2D' in item['type']:
            for postfix in weights_postfixes:
                weights = get_variables_containing(item['name']+'/'+postfix)
                if weights:
                    weights=weights[0]
                    break
            for postfix in biases_postfixes:
                biases = get_variables_containing(item['name']+'/'+postfix)
                if biases:
                    biases=biases[0]
                    break
            for postfix in activation_postfixes:
                activation = get_tensor_containing(item['name']+'/'+postfix)
                if activation is not None:
                    break
            node = find_tensors_containing(item['name']+'/'+item['type'])[0]
            input = get_tensor(node.input[0])
            output = activation #get_tensor_containing(item['name']+'/BiasAdd')
            strides = [int(a) for a in node.attr['strides'].list.i]
            padding = node.attr['padding'].s.decode()
            tf.logging.info('   Input: %s'%input)
            tf.logging.info('   Output: %s'%output)
            tf.logging.info('   Weights: %s'%weights)
            tf.logging.info('   Biases: %s'%biases)
            tf.logging.info('   Strides: %s'%strides)
            tf.logging.info('   Padding: %s'%padding)
            layer_collection.register_conv2d((weights,biases), strides, padding, input, output)
        elif 'MatMul' in item['type']:
            for postfix in weights_postfixes:
                weights = get_variables_containing(item['name']+'/'+postfix)
                if weights:
                    weights=weights[0]
                    break
            for postfix in biases_postfixes:
                biases = get_variables_containing(item['name']+'/'+postfix)
                if biases:
                    biases=biases[0]
                    break
            for postfix in activation_postfixes:
                activation = get_tensor_containing(item['name']+'/'+postfix)
                if activation is not None:
                    break
            node = find_tensors_containing(item['name']+'/'+item['type'])[0]
            input = get_tensor(node.input[0])
            output = activation #get_tensor_containing(item['name']+'/BiasAdd')
            tf.logging.info('   Input: %s'%input)
            tf.logging.info('   Output: %s'%output)
            tf.logging.info('   Weights: %s'%weights)
            tf.logging.info('   Biases: %s'%biases)
            layer_collection.register_fully_connected((weights,biases), input, output)
        #elif 'Prediction' in item['type']:
        #    node = find_tensors_containing(item['name'])[0]
        #    output = get_tensor_containing(item['name'])
        #    layer_collection.register_categorical_predictive_distribution(output)
        else:
            raise ValueError('kfac layer %s not recognized!'%item['name'])
        
    # register logits
    layer_collection.register_categorical_predictive_distribution(logits)
    tf.logging.info('K-FAC target: %s'%logits)
    return layer_collection


