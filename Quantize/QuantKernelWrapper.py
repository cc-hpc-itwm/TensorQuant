import tensorflow as tf
from tensorflow.python.framework import ops
import os

local_dir = os.path.dirname(__file__)

quant_module_log = tf.load_op_library(local_dir+'/../Kernels/QuantOp_log.so') 
quant_module_sparse = tf.load_op_library(local_dir+'/../Kernels/QuantOp_sparse.so') 
quant_module_halffp = tf.load_op_library(local_dir+'/../Kernels/QuantOp_halffp.so')
quant_module_binary = tf.load_op_library(local_dir+'/../Kernels/QuantOp_binary.so')
quant_module_ternary = tf.load_op_library(local_dir+'/../Kernels/QuantOp_ternary.so')

def quant_log(input1):
    '''Takes closest value of input to the form +/- 2^i.'''
    result = quant_module_log.quant_log(input1)
    return result
@ops.RegisterGradient("QuantLog")
def _quant_log_grad(op, grad):
  return [grad]

def quant_sparse(input1, threshold):
    '''Every element whose magnitude is below the threshold is set to 0.'''
    result = quant_module_sparse.quant_sparse(input1, threshold=threshold)
    return result
@ops.RegisterGradient("QuantSparse")
def _quant_sparse_grad(op, grad):
  return [grad]

def quant_halffp(input1):
    '''Rounds to half-precision floating point'''
    result = quant_module_halffp.quant_halffp(input1)
    return result
@ops.RegisterGradient("QuantHalffp")
def _quant_halffp_grad(op, grad):
  return [grad]

def quant_binary(input1, marginal):
    '''Binarizes to +/- marginal value'''
    result = quant_module_binary.quant_binary(input1, marginal)
    return result
@ops.RegisterGradient("QuantBinary")
def _quant_binary_grad(op, grad):
  return [grad]

def quant_ternary(input1, marginal, auto_threshold=True, threshold=0.5):
    '''Ternary quantization to +/- marginal and 0'''
    result = quant_module_ternary.quant_ternary(input1, marginal, auto_threshold, threshold)
    return result
@ops.RegisterGradient("QuantTernary")
def _quant_ternary_grad(op, grad):
  return [grad]
