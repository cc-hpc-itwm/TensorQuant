import tensorflow as tf

from tensorflow.python.layers import convolutional
from tensorflow.contrib.layers.python.layers import layers

import sys
sys.path.append('/home/loroch/TensorFlow/TensorLib')
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
from tensorflow.python.framework import tensor_shape
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
def conv2d(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
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
                use_quantized_weights = True):
  """ function call from slim library.
  """
  if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC', 'NCDHW']:
    raise ValueError('Invalid data_format: %r' % (data_format,))

  layer_variable_getter = layers._build_variable_getter(
      {'bias': 'biases', 'kernel': 'weights'})

  with variable_scope.variable_scope(
      scope, 'Conv', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)
    input_rank = inputs.get_shape().ndims


    if input_rank == 4:
      layer_class = QConv2D #convolutional.Conv2D
    else:
      raise ValueError('Convolution not supported for input with rank',
                       input_rank)

    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    layer = layer_class(filters=num_outputs,
                        kernel_size=kernel_size,
                        quantizer = quantizer,
                        use_quantized_weights=use_quantized_weights,
                        strides=stride,
                        padding=padding,
                        data_format=df,
                        dilation_rate=rate,
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
                        _reuse=reuse)
    outputs = layer.apply(inputs)

    # Add variables to collections.
    layers._add_variable_to_collections(layer.kernel, variables_collections, 'weights')
    if layer.use_bias:
      layers._add_variable_to_collections(layer.bias, variables_collections, 'biases')


    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
      if quantizer is not None:         # quantize after normalization
        outputs = quantizer.quantize(outputs)


    if activation_fn is not None:
      outputs = activation_fn(outputs)
      if quantizer is not None:         # quantize after activation
        outputs = quantizer.quantize(outputs)
    return slim_utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)


class _QConv(convolutional._Conv):
    """ Like _Conv, but with quantized convolution
    """
    def __init__(self, rank,
               filters,
               kernel_size,
               quantizer=None,      # quantizer object
               use_quantized_weights=True,   # use quantized weights by quantizer
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
        super(_QConv, self).__init__(rank,
               filters,
               kernel_size,
               strides,
               padding,
               data_format,
               dilation_rate,
               activation,
               use_bias,
               kernel_initializer,
               bias_initializer,
               kernel_regularizer,
               bias_regularizer,
               activity_regularizer,
               trainable,
               name,
               **kwargs)
        self.quantizer = quantizer
        self.use_quantized_weights = use_quantized_weights

    def build(self, input_shape):
        super(_QConv,self).build(input_shape)
        if self.quantizer is not None and self.use_quantized_weights:
            self.kernel = self.quantizer.quantize(self.kernel)


    def call(self, inputs):
        '''
        if self.use_quantized_weights:
                used_kernel = self.quantized_kernel
        else:
                used_kernel = self.kernel
        '''
        if self.rank == 2:          
          if self.quantizer is None:
            outputs = nn.convolution(
            input=inputs,
            filter=self.kernel,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=utils.convert_data_format(self.data_format, self.rank + 2))
          else: # with quantization
            outputs = q2dconvolution(input=inputs, filter=self.kernel, quantizer=self.quantizer,
                padding=self.padding.upper(), strides=self.strides, dilation_rate=self.dilation_rate,
                data_format=utils.convert_data_format(self.data_format, self.rank + 2))
        else:
          raise ValueError("quantized convolution not supported for input rank %d" % (self.rank))
        if self.bias is not None:
          if self.rank != 2 and self.data_format == 'channels_first':
            # bias_add does not support channels_first for non-4D inputs.
            if self.rank == 1:
              bias = array_ops.reshape(self.bias, (1, self.filters, 1))
            if self.rank == 3:
              bias = array_ops.reshape(self.bias, (1, self.filters, 1, 1))
            outputs += bias
          else:
            outputs = nn.bias_add(
                outputs,
                self.bias,
                data_format=utils.convert_data_format(self.data_format, 4))
            # Note that we passed rank=4 because bias_add will only accept
            # NHWC and NCWH even if the rank of the inputs is 3 or 5.
        if self.quantizer is not None:         # quantize after activation
            outputs = self.quantizer.quantize(outputs)
        if self.activation is not None:
          outputs = self.activation(outputs)
        return outputs

# called by slim
class QConv2D(_QConv):
  """ Quantized 2D convolution layer
  """

  def __init__(self, filters,
               kernel_size,
               quantizer = None,
               use_quantized_weights = True,
               strides=(1, 1),
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
    super(QConv2D, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        quantizer=quantizer,
        use_quantized_weights = use_quantized_weights,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name, **kwargs)


def q2dconvolution(input, filter, quantizer,  # pylint: disable=redefined-builtin
                padding, strides=None, dilation_rate=None,
                name=None, data_format=None):
  # pylint: disable=line-too-long
  """ quantized convolution"""
  # pylint: enable=line-too-long
  with ops.name_scope(name, "qconvolution", [input, filter]) as name:
    input = ops.convert_to_tensor(input, name="input")
    filter = ops.convert_to_tensor(filter, name="filter")
    num_total_dims = filter.get_shape().ndims
    if num_total_dims is None:
      num_total_dims = input.get_shape().ndims
    if num_total_dims is None:
      raise ValueError("rank of input or filter must be known")

    num_spatial_dims = num_total_dims - 2

    try:
      input.get_shape().with_rank(num_spatial_dims + 2)
    except ValueError:
      ValueError("input tensor must have rank %d" % (num_spatial_dims + 2))

    try:
      filter.get_shape().with_rank(num_spatial_dims + 2)
    except ValueError:
      ValueError("filter tensor must have rank %d" % (num_spatial_dims + 2))

    if data_format is None or not data_format.startswith("NC"):
      input_channels_dim = input.get_shape()[num_spatial_dims + 1]
      spatial_dims = range(1, num_spatial_dims+1)
    else:
      input_channels_dim = input.get_shape()[1]
      spatial_dims = range(2, num_spatial_dims+2)

    if not input_channels_dim.is_compatible_with(filter.get_shape()[
        num_spatial_dims]):
      raise ValueError(
          "number of input channels does not match corresponding dimension of filter, "
          "{} != {}".format(input_channels_dim, filter.get_shape()[
              num_spatial_dims]))
    #strides, dilation_rate = nn_ops._get_strides_and_dilation_rate(
    #    num_spatial_dims, strides, dilation_rate)
    strides = (1,strides[0],strides[1],1)
    output = q2dconvolution_op(input, filter, quantizer, strides, padding, data_format)
    return output


def q2dconvolution_op(inputs, filters, quantizer, strides, padding, data_format):
    ''' inputs:  [batch_size, image_height, image_width, input_channels] 
        filters: [filter_height, filter_width, input_channels, output_channels]
        quantizer: Quantizer object, has interface '.quantize(tensor)'       
    '''
    if data_format not in ("NHWC", None):
        raise ValueError("data_format other than NHWC not supported in quantized convolution, tried: %s"%(data_format))
    
    # split input batchwise
    batch_size = inputs.shape.dims[0].value
    output = tf.split(inputs,batch_size)

    # prepare filters
    filter_shape = filters.get_shape()
    #test_output = tf.extract_image_patches(output[0], 
    #                                   ksizes=(1,filters.shape.dims[0], filters.shape.dims[1],1), 
    #                                   strides=strides,
    #                                   rates=[1,1,1,1],
    #                                   padding=padding )
    #out_shape = (test_output.shape.dims[0].value,test_output.shape.dims[1].value,test_output.shape.dims[2].value,test_output.shape.dims[3].value)
    filters=tf.split(filters,filters.shape.dims[3].value,axis=3)
    
    for batch in range(batch_size):
        first = True
        #current_output = output[batch]
        #current_output = tf.reshape(current_output,[1,inputs.shape.dims[1].value,inputs.shape.dims[2].value,inputs.shape.dims[3].value])
        output_patch = tf.extract_image_patches(output[batch], 
                                           ksizes=(1,filter_shape.dims[0], filter_shape.dims[1],1), 
                                           strides=strides,
                                           rates=[1,1,1,1],
                                           padding=padding )
        for out_filter in range(filter_shape.dims[3].value):
            current_filter = filters[out_filter]
            current_filter = tf.reshape(current_filter, [1,1,1,output_patch.shape.dims[3].value])
            current_filter = tf.tile(current_filter,[1,output_patch.shape.dims[1].value,output_patch.shape.dims[2].value,1])  
            out = tf.multiply(output_patch,current_filter)
            if quantizer is not None:
                out = quantizer.quantize(out)     # quantize after multiply
            out = tf.reduce_sum(out,axis=3,keep_dims=True)
            if quantizer is not None:
                out = quantizer.quantize(out)     # quantize after add
            if first:
                outputs = out
                first=False
            else:
                outputs = tf.concat([outputs,out],3)
        output[batch] = outputs
    output = tf.squeeze(tf.stack(output),axis=[1])
    return output

