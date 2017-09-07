import tensorflow as tf
import numpy as np
bwreshape_module = tf.load_op_library('../Kernels/BWReshapeOp.so') 
round_module_zero = tf.load_op_library('../Kernels/RoundOp_zero.so') 
round_module_down = tf.load_op_library('../Kernels/RoundOp_down.so') 
round_module_nearest = tf.load_op_library('../Kernels/RoundOp_nearest.so') 
round_module_stochastic = tf.load_op_library('../Kernels/RoundOp_stochastic.so') 
fixedconv_module = tf.load_op_library('../Kernels/FixedConv.so') 

NEGATIVE_ROUND=False

def trunc(input1, fixed_size, fixed_prec):
    ''' Truncates input1 to a fixed point number. '''
    result_trunc = bwreshape_module.bitwidth_reshape(input1,fixed_size=fixed_size,fixed_prec=fixed_prec)
    return result_trunc

def round_zero(input1, fixed_size, fixed_prec):
    ''' Truncates input1 to a fixed point number. 
        Rounds towards zero.'''
    result = round_module_zero.round_zero(input1,fixed_size=fixed_size,fixed_prec=fixed_prec)
    return result

def round_down(input1, fixed_size, fixed_prec):
    ''' Truncates input1 to a fixed point number. 
        Rounds down towards negative numbers.'''
    result = round_module_down.round_down(input1,fixed_size=fixed_size,fixed_prec=fixed_prec)
    return result

def round_nearest(input1, fixed_size, fixed_prec):
    ''' Truncates input1 to a fixed point number. 
        Rounds towards the nearest number.'''
    result = round_module_nearest.round_nearest(input1,fixed_size=fixed_size,fixed_prec=fixed_prec)
    return result

def round_stochastic(input1, fixed_size, fixed_prec):
    ''' Truncates input1 to a fixed point number. 
        Rounds stochastically, whereas the fractional part influences the probability.'''
    result = round_module_stochastic.round_stochastic(input1,fixed_size=fixed_size,fixed_prec=fixed_prec)
    return result

def trunc2(input1, kernel, fixed_size, fixed_prec):
    ''' '''
    result = fixedconv_module.fixed_conv(input1, kernel, fixed_size=fixed_size,fixed_prec=fixed_prec)
    return result

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

def FixedPointOp(tensor, fixed_size, fixed_prec):
    fixed_max_signed = (2**(fixed_size-1)-1)/(2**fixed_prec)
    fixed_min_signed = -(2**(fixed_size-fixed_prec-1))

    if NEGATIVE_ROUND:
        # rounds towards zero (e.g. -0.001 -> 0 )
	    tensor = tf.floor(tf.abs(tensor)*(2**fixed_prec)) / (2**fixed_prec) * tf.sign(tensor);
    else:
        # rounds towards negative (e.g. -0.001 -> -0.5 )
        tensor = tf.floor(tensor*(2**fixed_prec)) / (2**fixed_prec);

    tensor = tf.maximum(tf.minimum(tensor,tf.ones(tensor.shape)*fixed_max_signed), tf.ones(tensor.shape)*fixed_min_signed);
    return tensor
