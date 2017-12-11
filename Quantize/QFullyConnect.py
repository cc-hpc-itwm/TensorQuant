# File copied from tensorflow/contrib/layers/python/layers/layers.py
# Minor modifications are applied to enable intrinsic quantization

import tensorflow as tf

import functools
import six

from tensorflow.python.layers import convolutional
from tensorflow.contrib.layers.python.layers import layers

from Quantize import Quantizers

# from slim
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils as slim_utils
from tensorflow.python.layers import utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import  normalization as normalization_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages


# from slim: tensorflow/contrib/layers/python/layers/layers.py
@add_arg_scope
def fully_connected(inputs,
                    num_outputs,
                    activation_fn=nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=None,
                    biases_initializer=init_ops.zeros_initializer(),
                    biases_regularizer=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None,
                    quantizer=None,
                    weight_quantizer=None):
  """ """
  if not isinstance(num_outputs, six.integer_types):
    raise ValueError(
        'num_outputs should be int or long, got %s.' % (num_outputs,))

  layer_variable_getter = layers._build_variable_getter({'bias': 'biases',
                                                  'kernel': 'weights'})

  with variable_scope.variable_scope(
      scope, 'fully_connected', [inputs],
      reuse=reuse, custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)
    layer = QDense(
        units=num_outputs,
        activation=None,
        use_bias=not normalizer_fn and biases_initializer,
        kernel_initializer=weights_initializer,
        bias_initializer=biases_initializer,
        kernel_regularizer=weights_regularizer,
        bias_regularizer=biases_regularizer,
        activity_regularizer=None,
        trainable=trainable,
        name=sc.name,
        dtype=inputs.dtype.base_dtype,
        _scope=sc,
        _reuse=reuse,
        quantizer=quantizer,
        weight_quantizer=weight_quantizer)
    outputs = layer.apply(inputs)

    # Add variables to collections.
    layers._add_variable_to_collections(layer.kernel, variables_collections, 'weights')
    if layer.bias is not None:
      layers._add_variable_to_collections(layer.bias, variables_collections, 'biases')

    # Apply normalizer function / layer.
    if normalizer_fn is not None:
      if not normalizer_params:
        normalizer_params = {}
      outputs = normalizer_fn(outputs, **normalizer_params)
      if quantizer is not None:
        outputs = quantizer.quantize(outputs)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
      if quantizer is not None:
        outputs = quantizer.quantize(outputs)

    return slim_utils.collect_named_outputs(
        outputs_collections, sc.original_name_scope, outputs)

class QDense(core_layers.Dense):
  """
  """

  def __init__(self, units,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               quantizer=None,
               weight_quantizer=None,
               **kwargs):
    super(QDense, self).__init__(units=units,
               activation=activation,
               use_bias=use_bias,
               kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               activity_regularizer=activity_regularizer,
               trainable=trainable,
               name=name,
               **kwargs)
    self.quantizer = quantizer
    self.weight_quantizer = weight_quantizer

  # overridden call method
  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    shape = inputs.get_shape().as_list()
    output_shape = shape[:-1] + [self.units]

    # quantize the weights, if there is an weight quantizer
    if self.weight_quantizer is not None:
      with tf.variable_scope("quant_weights"):
        used_kernel = self.weight_quantizer.quantize(self.kernel)
      with tf.variable_scope("quant_biases"):
        used_bias = self.weight_quantizer.quantize(self.bias)
    else:
        used_kernel = self.kernel
        used_bias = self.bias
    # if intrinsic quantization, apply intr. quantization to weights, too!
    if self.quantizer is not None:
        used_kernel = self.quantizer.quantize(used_kernel)
        used_bias = self.quantizer.quantize(used_bias)

    if len(output_shape) > 2:
      ## Broadcasting is required for the inputs.
      #outputs = standard_ops.tensordot(inputs, self.kernel, [[len(shape) - 1],
      #                                                       [0]])
      ## Reshape the output back to the original ndim of the input.
      #outputs.set_shape(output_shape)
      raise ValueError('output_shape > 2 not supported for quantized operation, tried $d.' %(len(output_shape))) 
    else:
      if self.quantizer is None:
        outputs = standard_ops.matmul(inputs, used_kernel)
      else: # with quantization
        outputs = qmatmul(inputs, used_kernel, self.quantizer)
    #TODO: quantize after bias and activation
    if self.use_bias:
      outputs = nn.bias_add(outputs, used_bias)
      if self.quantizer is not None:
        outputs = self.quantizer.quantize(outputs)
    if self.activation is not None:
      # never called, since activation performed in upper hierarchy
      outputs= self.activation(outputs)  # pylint: disable=not-callable
      if self.quantizer is not None:
        outputs = self.quantizer.quantize(outputs)
    return outputs

def qmatmul(inputs, kernel, quantizer):
    batch_size = inputs.shape.dims[0].value
    input_size = inputs.shape.dims[1].value
    output_size = kernel.get_shape().dims[1].value
    output = tf.split(inputs,batch_size)
    for batch in range(batch_size):
        current_output = output[batch]
        current_output = tf.reshape(current_output,[input_size,1])
        current_output = tf.tile(current_output,[1,output_size])
        current_output = tf.multiply(current_output,kernel)
        if quantizer is not None:
            current_output = quantizer.quantize(current_output)     # quantize after multiply
        current_output = tf.reduce_sum(current_output,axis=[0])
        if quantizer is not None:
            current_output = quantizer.quantize(current_output)     # quantize after add
        output[batch] = current_output
    output = tf.stack(output)
    return output


