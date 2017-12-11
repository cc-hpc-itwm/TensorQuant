import tensorflow as tf
import numpy as np
import Quantizers
import math

from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn

input_width = input_height = 4
batch_size = 2
input_channels = 2
testdata_scale=10

threshold = 5

def check(val):
    if val>0:
        print('---failed!---')
    else:
        print('+++passed!+++')

def log2(tensor):
    signs=tf.sign(tensor)
    tensor= tf.floor(tf.log(tf.abs(tensor))/math.log(2))
    tensor= tf.pow(2*tf.ones_like(tensor),tensor)
    tensor= tensor*signs
    return tensor

def sparse(tensor, threshold):
    tensor = tensor*tf.to_float(tf.abs(tensor)>threshold)
    return tensor

inputs_vals = np.random.normal(size=(batch_size,input_width,input_height,input_channels))*testdata_scale
inputs = tf.constant(inputs_vals,dtype=tf.float32)
#inputs = tf.zeros_like(inputs)

quantizer_log = Quantizers.LogarithmicQuantizer()
output_log = quantizer_log.quantize(inputs)
gold_output_log = log2(inputs)

quantizer_sparse = Quantizers.SparseQuantizer(threshold)
output_sparse = quantizer_sparse.quantize(inputs)
gold_output_sparse = sparse(inputs, threshold)

with tf.Session() as sess:
  '''
  print('input:')
  print(sess.run(inputs))
  print('quantized:')
  print(sess.run(output_log))
  print('gold quantized:')
  print(sess.run(gold_output_log))
  '''
  result_log=np.sum(np.absolute(gold_output_log.eval().flatten()-output_log.eval().flatten()))
  result_sparse=np.sum(np.absolute(gold_output_sparse.eval().flatten()-output_sparse.eval().flatten()))

print('LogQuantizer test:')
check(result_log)

print('SparseQuantizer test:')
check(result_sparse)

