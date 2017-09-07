import tensorflow as tf
import numpy as np
import sys
sys.path.append('../../TensorLib')
import Quantizers
import FixedPoint as FP

from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn

input_width = input_height = 3
batch_size = 1
input_channels = 1
# output_channels = 6
#kernel_size = 3
#strides = 2
#padding='VALID'
fixed_size = 8
fixed_prec = 3
testdata_scale=10

#inputs_vals = np.ones((batch_size,input_width*input_height*input_channels))
#inputs_vals = np.arange(input_width*input_height*input_channels*batch_size).reshape(batch_size,input_width*input_height*input_channels)
inputs_vals = np.random.normal(size=(batch_size,input_width,input_height,input_channels))*testdata_scale
#print(inputs_vals)
#filters_vals = np.ones((input_channels*input_width*input_height,output_channels)) 
#filters_vals = np.arange(input_channels*input_width*input_height*output_channels).reshape(input_channels*input_width*input_height,output_channels)


inputs = tf.constant(inputs_vals,dtype=tf.float32)

#quantizer = Quantizers.FixedPointQuantizer(fixed_size,fixed_prec)
quantizer = Quantizers.FixedPointQuantizer_stochastic(fixed_size,fixed_prec)
gold_output = quantizer.quantize(inputs)
output = FP.FixedPointOp(inputs,fixed_size,fixed_prec)

with tf.Session() as sess:
  gold_result=gold_output.eval().flatten()
  result=output.eval().flatten()
  print('input:')
  print(sess.run(inputs))
  print('output:')
  print(sess.run(gold_output))
  #print(sess.run(output))
  pass

failed=False
for i in range(len(result)):
    if result[i] != gold_result[i]:
        failed = True
        break
print('FixedPoint test:')
if failed:
    print('---failed!---')
else:
    print('+++passed!+++')


