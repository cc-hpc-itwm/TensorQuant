import tensorflow as tf
from tensorflow.python.framework import ops

quant_module_log = tf.load_op_library('../Kernels/QuantOp_log.so') 
quant_module_sparse = tf.load_op_library('../Kernels/QuantOp_sparse.so') 

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
