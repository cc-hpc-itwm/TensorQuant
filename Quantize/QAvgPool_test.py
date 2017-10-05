import tensorflow as tf
import numpy as np
import Quantizers
import QAvgPool as QAP

from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn

input_width = input_height = 28
batch_size = 10
input_channels = 8
kernel_size = 3
strides = 2
padding='VALID'
fixed_size = 8
fixed_prec = 4
testdata_scale=10

inputs_vals = np.random.normal(size=(batch_size,input_width,input_height,input_channels))*testdata_scale//1

inputs = tf.constant(inputs_vals,dtype=tf.float32)

quantizer = Quantizers.NoQuantizer()
output = QAP.avg_pool2d(inputs,kernel_size,strides,padding=padding, quantizer=quantizer)
gold_output = nn.avg_pool(inputs,[1,kernel_size,kernel_size,1],[1,strides,strides,1],padding=padding)

with tf.Session() as sess:
  gold_result=gold_output.eval().flatten()
  result=output.eval().flatten()
  # print(sess.run(gold_output))
  pass

failed=False
for i in range(len(result)):
    if result[i] != gold_result[i]:
        failed = True
        break
print('QAvgPool test:')
if failed:
    print('---failed!---')
else:
    print('+++passed!+++')


