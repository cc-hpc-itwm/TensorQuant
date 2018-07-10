# Utilities used in the evaluation and training python scripts.
#
# author: Dominik Loroch
# date: August 2017

import sys
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import re
import os


slim = tf.contrib.slim

def heatmap_conv(kernel, pad = 1):
    """Visualize conv features as an image (mostly for the 1st layer).
    Places kernel into a grid, with some paddings between adjacent filters.

    see: gist.github.com/kukuruza/03731dc494603ceab0c5

    Args:
        kernel: tensor of shape [height, width, NumChannels, NumKernels]     
        pad: number of black pixels around each filter (between them)

    Return:
        Tensor of shape [batch=NumChannels,
                (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels=3].
    """

    def factorization(n):
        for i in range(int(math.sqrt(float(n))), 0, -1):
            if n % i == 0:
                return (i, int(n / i))

    (grid_Y, grid_X) = factorization (kernel.get_shape().dims[3].value)
    # scale weights to [-1,1]
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
    """Visualize fc-layer as an image.

    Args:
        kernel: tensor of shape [kernel_inputs, kernel_outputs]   
        pad: number of black pixels around filter

    Return:
        Tensor of shape [batch=1, kernel_inputs, kernel_outputs, NumChannels=3].
    """

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


def heatmap_activation(kernel, pad = 1):
    """Visualize fc-layer as an image.

    Args:
        kernel: tensor of shape [batch, height, width, channels]   
        pad: number of black pixels around filter

    Return:
        Tensor of shape [ batch,
                (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, 3].
    """

    def factorization(n):
        for i in range(int(math.sqrt(float(n))), 0, -1):
            if n % i == 0:
                return (i, int(n / i))

    (grid_Y, grid_X) = factorization (kernel.get_shape().dims[3].value)
    # scale weights to [-1,1]
    x_max = tf.reduce_max(tf.abs(kernel))
    kernel = (kernel) / (x_max)
    kernel_batch = kernel.get_shape().dims[0].value
    kernel_height = kernel.get_shape().dims[1].value
    kernel_width = kernel.get_shape().dims[2].value
    kernel_channels = kernel.get_shape().dims[3].value
    #kernel shape: [batch, height, width, channels]

    red_channel = (tf.ones_like(kernel)-tf.to_float(kernel>0)*kernel)
    green_channel = (tf.ones_like(kernel)-tf.abs(kernel))
    blue_channel = (tf.ones_like(kernel)+tf.to_float(kernel<0)*kernel)
    kernel = tf.transpose(tf.stack([red_channel,green_channel,blue_channel]), (4,2,3,1,0))
    #kernel shape: [channels, height, width, batch, rgb]
    # channels will be the batch, rgb are the color channels. 
    # channels will be reduced to single image

    # pad X and Y
    kernel = tf.pad(kernel, tf.constant( 
    [[0,0],[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')
    tile_height= kernel.get_shape().dims[1].value
    tile_width= kernel.get_shape().dims[2].value
    # 2*pad added to height and width

    # organize grid on Y axis
    kernel = tf.reshape(kernel, tf.stack([grid_X, tile_height * grid_Y, 
                                        tile_width, kernel_batch, 3]))
    # switch X and Y axes
    kernel = tf.transpose(kernel, (0, 2, 1, 3, 4))
    #kernel shape: [channels, width, height, batch, rgb]
    # organize grid on X axis, drop 5th dimension
    kernel = tf.reshape(kernel, tf.stack([tile_width * grid_X, tile_height * grid_Y, 
                                            kernel_batch, 3]))
    kernel = tf.transpose(kernel, (2, 1, 0, 3))
    #kernel shape: [batch, height, width, rgb]
    return kernel
