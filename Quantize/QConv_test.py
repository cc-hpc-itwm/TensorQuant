"""
Test for QConv implementation.
Tests the convolution reimplementation against the TensorFlow nn.convolution implementation, and the atrous_conv2d reimplementation against nn.atrous_conv2d.
The input is a tensor [batch, width, height, channels] filled with numbers from zero counting up. Weight values are filled with numbers from zero counting up as well.
Test passes if reimplementation and TF version have the same output.
"""

import tensorflow as tf
import numpy as np
import Quantizers
import QConv

from tensorflow.python.ops import nn

input_width = input_height = 10
filter_width = filter_height = 3
batch_size = 2
input_channels = 2
output_channels = 2
stride=(2,2)
rate=(1,1)
padding = "VALID" # "VALID" or "SAME"
data_format = "NHWC"
fixed_size = 32
fixed_prec = 8

testdata_scale = 10

inputs_vals = np.arange(input_width*input_height*input_channels*batch_size).reshape(batch_size,input_width,input_height,input_channels)
filters_vals = np.arange(filter_width*filter_height*input_channels*output_channels).reshape(filter_width,filter_height,input_channels,output_channels)


inputs = tf.constant(inputs_vals,dtype=tf.float32)
filters = tf.constant(filters_vals,dtype=tf.float32)

quantizer = Quantizers.NoQuantizer()

results={} # dictionary with test results

test_name='conv2d'
results[test_name]={}
results[test_name]['quant'] = QConv.q2dconvolution(input=inputs, filter=filters, quantizer=quantizer, 
                                strides=stride, dilation_rate=rate, 
                                padding=padding, data_format=data_format)
results[test_name]['gold'] = nn.convolution(
                                input=inputs,
                                filter=filters,
                                dilation_rate=rate,
                                strides=stride,
                                padding=padding)

test_name='atrous_conv2d'
results[test_name]={}
results[test_name]['quant'] = QConv.atrous_conv2d(inputs, filters, rate, padding, 
                                quantizer=quantizer)
results[test_name]['gold'] = nn.atrous_conv2d(inputs, filters, rate, padding)


with tf.Session() as sess:
    for key in results.keys():
        results[key]['quant']=results[key]['quant'].eval().flatten()
        results[key]['gold']=results[key]['gold'].eval().flatten()
        #print(key)
        #print(results[key]['quant'])
        #print('--------------------------------')
        #print(results[key]['gold'])
        #print('################################')


for key in results.keys():
    if any(results[key]['quant'] != results[key]['gold']):
        results[key]['failed'] = True
    else:
        results[key]['failed'] = False

for key in results.keys():
    if results[key]['failed']:
        print('%s: ---failed!---'%key)
    else:
        print('%s: +++passed!+++'%key)



