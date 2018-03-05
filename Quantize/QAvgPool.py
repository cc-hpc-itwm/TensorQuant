# File copied from tensorflow/python/layers/pooling.py
# Minor modifications are applied to enable intrinsic quantization

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils as slim_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import constant_op
from tensorflow.python.layers import base
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DATA_FORMAT_NCDHW = 'NCDHW'
DATA_FORMAT_NDHWC = 'NDHWC'


@add_arg_scope
def avg_pool2d(inputs,
               kernel_size,
               stride=2,
               padding='VALID',
               data_format=DATA_FORMAT_NHWC,
               outputs_collections=None,
               scope=None,
               quantizer=None):
  """Adds a 2D average pooling op.
  It is assumed that the pooling is done per image but not in batch or channels.
  Args:
    inputs: A 4-D tensor of shape `[batch_size, height, width, channels]` if
      `data_format` is `NHWC`, and `[batch_size, channels, height, width]` if
      `data_format` is `NCHW`.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of the
      pooling kernel over which the op is computed. Can be an int if both
      values are the same.
    stride: A list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same. Note that presently
      both strides must have the same value.
    padding: The padding method, either 'VALID' or 'SAME'.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    outputs_collections: The collections to which the outputs are added.
    scope: Optional scope for name_scope.
  Returns:
    A `Tensor` representing the results of the pooling operation.
  Raises:
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
  """
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')
  with ops.name_scope(scope, 'AvgPool2D', [inputs]) as sc:
    inputs = ops.convert_to_tensor(inputs)
    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    layer = AveragePooling2D(pool_size=kernel_size,
                                            strides=stride,
                                            padding=padding,
                                            data_format=df,
                                            _scope=sc,
                                            quantizer=quantizer)
    outputs = layer.apply(inputs)
    return slim_utils.collect_named_outputs(outputs_collections, sc, outputs)


# tensorflow/python/layers/pooling.py
class _Pooling2D(base.Layer):
  """Pooling layer for arbitrary pooling functions, for 2D inputs (e.g. images).
  This class only exists for code reuse. It will never be an exposed API.
  Arguments:
    pool_function: The pooling function to apply, e.g. `tf.nn.max_pool`.
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_function, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, quantizer=None, **kwargs):
    super(_Pooling2D, self).__init__(name=name, **kwargs)
    self.pool_function = pool_function
    self.pool_size = utils.normalize_tuple(pool_size, 2, 'pool_size')
    self.strides = utils.normalize_tuple(strides, 2, 'strides')
    self.padding = utils.normalize_padding(padding)
    self.data_format = utils.normalize_data_format(data_format)
    self.input_spec = base.InputSpec(ndim=4)
    self.quantizer=quantizer

  def call(self, inputs):
    if self.data_format == 'channels_last':
      pool_shape = (1,) + self.pool_size + (1,)
      strides = (1,) + self.strides + (1,)
    else:
      pool_shape = (1, 1) + self.pool_size
      strides = (1, 1) + self.strides
    if self.quantizer is None:
      outputs = self.pool_function(
        inputs,
        ksize=pool_shape,
        strides=strides,
        padding=self.padding.upper(),
        data_format=utils.convert_data_format(self.data_format, 4))
    else:
        outputs = self.pool_function(
        inputs,
        ksize=pool_shape,
        strides=strides,
        padding=self.padding.upper(),
        quantizer=self.quantizer,
        data_format=utils.convert_data_format(self.data_format, 4))
    return outputs


# tensorflow/python/layers/pooling.py
class AveragePooling2D(_Pooling2D):
  """Average pooling layer for 2D inputs (e.g. images).
  Arguments:
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, quantizer=None, **kwargs):
    if quantizer is None:
        pooling_fn = nn.avg_pool
    else:
        pooling_fn = avg_pool
    super(AveragePooling2D, self).__init__(
        pooling_fn,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, name=name, quantizer=quantizer, **kwargs)



def avg_pool(value, ksize, strides, padding, quantizer=None, data_format="NHWC", name=None):
  """Performs the average pooling on the input (quantized version).
  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.
  Args:
    value: A 4-D `Tensor` of shape `[batch, height, width, channels]` and type
      `float32`, `float64`, `qint8`, `quint8`, or `qint32`.
    ksize: A list of ints that has length >= 4.
      The size of the window for each dimension of the input tensor.
    strides: A list of ints that has length >= 4.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the @{tf.nn.convolution$comment here}
    quantizer: The quantizer which is applied after every step.
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    name: Optional name for the operation.
  Returns:
    A `Tensor` with the same type as `value`.  The average pooled output tensor.
  """
  kelems = ksize[1]*ksize[2]
  with ops.name_scope(name, "AvgPool", [value]) as name:
    value = ops.convert_to_tensor(value, name="input")
    output = array_ops.extract_image_patches(value,ksize,strides,[1,1,1,1],padding)
    output = array_ops.reshape(output,[output.shape.dims[0].value, 
                      output.shape.dims[1].value, output.shape.dims[2].value, 
                      kelems, value.shape.dims[3].value])
    output = math_ops.reduce_sum(output,axis=3)
    output = quantizer.quantize(output)
    output = output / kelems
    output = quantizer.quantize(output)
    return output
