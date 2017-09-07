import tensorflow as tf
import numpy as np

import Quantizers
import QConv

from tensorflow.python.ops import nn

input_width = input_height = 20
filter_width = filter_height = 3
batch_size = 10
input_channels = 20
output_channels = 20
stride=2
strides = [1,stride,stride,1]
padding = "SAME" # "VALID" or "SAME"
data_format = "NHWC"
fixed_size = 32
fixed_prec = 8

testdata_scale = 10

#inputs_vals = np.ones((batch_size,input_width,input_height,input_channels)) # batch, width, height, channels
#inputs_vals = np.tile(np.repeat(np.arange(1,10),input_channels).reshape((3,3,input_channels)),(batch_size,1,1,1))
#inputs_vals = np.arange(input_width*input_height*input_channels*batch_size).reshape(batch_size,input_width,input_height,input_channels)
inputs_vals = np.random.normal(size=(batch_size,input_width,input_height,input_channels))*testdata_scale//1
#print(inputs_vals)
#filters_vals = np.ones((filter_width,filter_height,input_channels,output_channels)) # width, height, in_channels, out_channels
#filters_vals = np.arange(filter_width*filter_height*input_channels*output_channels).reshape(filter_width,filter_height,input_channels,output_channels)
filters_vals = np.random.normal(size=(filter_width,filter_height,input_channels,output_channels))*testdata_scale//1

inputs = tf.constant(inputs_vals,dtype=tf.float32)
filters = tf.constant(filters_vals,dtype=tf.float32)

#quantizer = Quantizers.FixedPointQuantizer(fixed_size,fixed_prec)
quantizer = Quantizers.NoQuantizer()
output = QConv.q2dconvolution_op(inputs, filters, quantizer, strides, padding, data_format)
#output = QConv.q2dconvolutionfixed_op(inputs, filters, fixed_size, fixed_prec, strides, padding, data_format)

gold_output = nn.convolution(
        input=inputs,
        filter=filters,
        dilation_rate=(1,1),
        strides=(stride,stride),
        padding=padding)

with tf.Session() as sess:
  gold_result=gold_output.eval().flatten()
  result=output.eval().flatten()
  #print(sess.run(gold_output))
  #print('################################')
  #print(sess.run(output))
  pass

failed=False
for i in range(len(result)):
    if result[i] != gold_result[i]:
        failed = True
        break

print('QConv test:')
if failed:
    print('---failed!---')
else:
    print('+++passed!+++')



