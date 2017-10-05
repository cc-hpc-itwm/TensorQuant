import tensorflow as tf
import numpy as np
import Quantizers
import FixedPoint as FP

from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn

input_width = input_height = 3
batch_size = 1
input_channels = 1
fixed_size = 8
fixed_prec = 3
testdata_scale=10


inputs_vals = np.random.normal(size=(batch_size,input_width,input_height,input_channels))*testdata_scale

inputs = tf.constant(inputs_vals,dtype=tf.float32)

quantizer = Quantizers.FixedPointQuantizer_down(fixed_size,fixed_prec)
output = quantizer.quantize(inputs)
gold_output = FP.FixedPointOp(inputs,fixed_size,fixed_prec)

with tf.Session() as sess:
  gold_result=gold_output.eval().flatten()
  result=output.eval().flatten()
  print('input:')
  print(sess.run(inputs))
  print('quantized input:')
  print(sess.run(output))
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


