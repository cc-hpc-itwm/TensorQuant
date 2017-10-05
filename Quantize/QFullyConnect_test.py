import tensorflow as tf
import numpy as np
import Quantizers
import QFullyConnect as QFC

from tensorflow.python.ops import standard_ops

input_width = input_height = 28
batch_size = 10
input_channels = 8
output_channels = 6
fixed_size = 8
fixed_prec = 4


inputs_vals = np.arange(input_width*input_height*input_channels*batch_size).reshape(batch_size,input_width*input_height*input_channels)
filters_vals = np.arange(input_channels*input_width*input_height*output_channels).reshape(input_channels*input_width*input_height,output_channels)

inputs = tf.constant(inputs_vals,dtype=tf.float64)
filters = tf.constant(filters_vals,dtype=tf.float64)

quantizer = Quantizers.NoQuantizer()
output = QFC.qmatmul(inputs, filters, quantizer)
gold_output = standard_ops.matmul(inputs, filters)

with tf.Session() as sess:
  gold_result=gold_output.eval().flatten()
  result=output.eval().flatten()
  #print(sess.run(gold_output))
  #print(sess.run(filters))
  pass

failed=False
for i in range(len(result)):
    if result[i] != gold_result[i]:
        failed = True
        break

print('QFullyConnect test:')
if failed:
    print('---failed!---')
else:
    print('+++passed!+++')


