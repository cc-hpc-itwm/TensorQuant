import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import os

local_dir = os.path.dirname(__file__)

round_module_zero = tf.load_op_library(local_dir+'/../Kernels/RoundOp_zero.so')
round_module_down = tf.load_op_library(local_dir+'/../Kernels/RoundOp_down.so') 
round_module_nearest = tf.load_op_library(local_dir+'/../Kernels/RoundOp_nearest.so') 
round_module_stochastic = tf.load_op_library(local_dir+'/../Kernels/RoundOp_stochastic.so') 

def round_zero(input1, fixed_size, fixed_prec):
    ''' Truncates input1 to a fixed point number. 
        Rounds towards zero.'''
    result = round_module_zero.round_zero(input1,fixed_size=fixed_size,fixed_prec=fixed_prec)
    return result
@ops.RegisterGradient("RoundZero")
def _round_zero_grad(op, grad):
  return [grad]

def round_down(input1, fixed_size, fixed_prec):
    ''' Truncates input1 to a fixed point number. 
        Rounds down towards negative numbers.'''
    result = round_module_down.round_down(input1,fixed_size=fixed_size,fixed_prec=fixed_prec)
    return result
@ops.RegisterGradient("RoundDown")
def _round_down_grad(op, grad):
  return [grad]

def round_nearest(input1, fixed_size, fixed_prec):
    ''' Truncates input1 to a fixed point number. 
        Rounds towards the nearest number.'''
    result = round_module_nearest.round_nearest(input1,fixed_size=fixed_size,fixed_prec=fixed_prec)
    return result
@ops.RegisterGradient("RoundNearest")
def _round_nearest_grad(op, grad):
  return [grad]

def round_stochastic(input1, fixed_size, fixed_prec):
    ''' Truncates input1 to a fixed point number. 
        Rounds randomly, but the fractional part influences the probability.'''
    result = round_module_stochastic.round_stochastic(input1,fixed_size=fixed_size,fixed_prec=fixed_prec)
    return result
@ops.RegisterGradient("RoundStochastic")
def _round_stochastic_grad(op, grad):
  return [grad]


# Fixed point functions, python implementation
# -------------------------------------------------
# for debugging
def toFixed(fp_number, fixed_size, fixed_prec):
    ''' Turns the elements of a floating point numpy matrix into fixed point equivalents with bitwidth fixed_size and fractional bits fixed_prec.'''
    fixed_max_signed = (2.0**(fixed_size-1)-1)/(2**fixed_prec)  # maximum value of fixed representation
    fixed_min_signed = -(2.0**(fixed_size-fixed_prec-1))        # minimum (negative) value of fixed representation
    # adjust fractional part (round towards zero)
    fixed_number = np.multiply( ((np.absolute(fp_number)*2**fixed_prec)//1) /2**fixed_prec , (np.sign(fp_number)) )  
    # handle overflow (saturate number towards maximum or minimum)    
    fixed_number = np.maximum(np.minimum(fixed_number,fixed_max_signed), fixed_min_signed)
    return fixed_number

def fixTensor(tensor, session, fixed_size, fixed_prec):
    ''' Truncates elements of a tensor to fixed point representation. '''
    tensor_a = session.run(tensor)
    tensor_a = toFixed(tensor_a, fixed_size, fixed_prec)
    tensor.load(tensor_a,session)
    return tensor

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
