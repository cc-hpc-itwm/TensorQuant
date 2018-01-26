import tensorflow as tf
import numpy as np
import Quantizers
import math
import time
import struct


from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn

hex_lambda = lambda x : hex(struct.unpack('<I', struct.pack('<f', x))[0])
toHex = np.vectorize(hex_lambda)

input_width = input_height = 4
batch_size = 2
input_channels = 2
entries = input_width*input_height*batch_size*input_channels
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

def halffp(tensor):
    tensor = tf.cast(tensor,tf.float16)
    return tensor

inputs_vals = np.random.normal(size=(batch_size,input_width,input_height,input_channels))*testdata_scale
#inputs_vals = 1.0 - 2.0**-np.arange(entries)
inputs = tf.constant(inputs_vals,dtype=tf.float32)
#inputs = tf.zeros_like(inputs)

quantizer_log = Quantizers.LogarithmicQuantizer()
output_log = quantizer_log.quantize(inputs)
gold_output_log = log2(inputs)

quantizer_sparse = Quantizers.SparseQuantizer(threshold)
output_sparse = quantizer_sparse.quantize(inputs)
gold_output_sparse = sparse(inputs, threshold)

quantizer_halffp = Quantizers.HalffpQuantizer()
output_halffp = quantizer_halffp.quantize(inputs)
gold_output_halffp = halffp(inputs)

with tf.Session() as sess:
  '''
  print('input:')
  print(toHex(sess.run(inputs)))
  print('quantized:')
  print(toHex(sess.run(output_halffp)))
  print('gold quantized:')
  print(toHex(sess.run(gold_output_halffp)))
  '''
  
  result_log=np.sum(
            np.absolute(gold_output_log.eval().flatten()-output_log.eval().flatten()))
  result_sparse=np.sum(
            np.absolute(gold_output_sparse.eval().flatten()-output_sparse.eval().flatten()))
  
  # rounding in TF FP16 format is different than in Halffp kernel implementation!
  # mantissa is rounded up in TF FP16 and cut in kernel.
  # rounding bit in integer representation has value 8192, difference between TF FP16 and
  # kernel is 0 or that number.
  gold_output_halffp = np.array(
        [struct.unpack('<I', struct.pack('<f', x))[0] 
         for x in gold_output_halffp.eval().flatten()]) 
  output_halffp = np.array(
        [struct.unpack('<I', struct.pack('<f', x))[0] 
         for x in output_halffp.eval().flatten()])  
  result_halffp = gold_output_halffp-output_halffp
  result_halffp = np.absolute( 
                np.sum( (result_halffp==0).astype(np.int32) 
                        + (result_halffp==8192).astype(np.int32) )
                - result_halffp.size)

  #result_halffp=np.sum(np.absolute(gold_output_halffp.eval().flatten()-output_halffp.eval().flatten()))

  '''
  start=time.time()
  for i in range(100000):
    output_halffp.eval()
  runtime = time.time()-start
  print('kernel-version time: %fs'%runtime)

  start=time.time()
  for i in range(100000):
    gold_output_halffp.eval()
  runtime = time.time()-start
  print('tf-version time: %fs'%runtime)
  '''

print('LogQuantizer test:')
check(result_log)

print('SparseQuantizer test:')
check(result_sparse)

print('HalffpQuantizer test:')
check(result_halffp)

