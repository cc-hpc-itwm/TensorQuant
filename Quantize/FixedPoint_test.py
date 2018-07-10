"""
Test for FixedPoint implementation.
Tests the round down kernel implementation against a python implementation.
The input to both implementations is a random tensor [batch, width, height, channels].
Test passes if both implementations have the same output.
"""

import tensorflow as tf
import numpy as np
import Quantizers

from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn

ZERO_ROUND=False
def FixedPointOp(tensor, fixed_size, fixed_prec):
    fixed_max_signed = (2**(fixed_size-1)-1)/(2**fixed_prec)
    fixed_min_signed = -(2**(fixed_size-fixed_prec-1))

    if ZERO_ROUND:
        # rounds towards zero (e.g. -0.001 -> 0 )
	    tensor = tf.floor(tf.abs(tensor)*(2**fixed_prec)) / (2**fixed_prec) * tf.sign(tensor);
    else:
        # rounds towards negative (e.g. -0.001 -> -0.5 )
        tensor = tf.floor(tensor*(2**fixed_prec)) / (2**fixed_prec);

    tensor = tf.maximum(tf.minimum(tensor,tf.ones(tensor.shape)*fixed_max_signed), tf.ones(tensor.shape)*fixed_min_signed);
    return tensor


input_width = input_height = 3
batch_size = 1
input_channels = 1
fixed_size = 8
fixed_prec = 0
testdata_scale=1


inputs_vals = np.random.normal(size=(batch_size, input_width, input_height, input_channels)) * testdata_scale


inputs = tf.constant(inputs_vals,dtype=tf.float32)

quantizer = Quantizers.FixedPointQuantizer_down(fixed_size,fixed_prec)
output = quantizer.quantize(inputs)
gold_output = FixedPointOp(inputs,fixed_size,fixed_prec)

with tf.Session() as sess:
  gold_result=gold_output.eval().flatten()
  result=output.eval().flatten()
  #print('input:')
  #print(sess.run(inputs))
  #print('quantized input:')
  #print(sess.run(output))

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


