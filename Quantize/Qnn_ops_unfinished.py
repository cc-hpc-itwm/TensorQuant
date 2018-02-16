
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrappers for primitive Neural Net (NN) Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import numpy as np

#from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
# original file
from tensorflow.python.ops import nn_ops

# Conv reimplementation
import QConv

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *
# pylint: enable=wildcard-import

from tensorflow.python.util import deprecation
#from tensorflow.python.util.tf_export import tf_export

# Aliases for some automatically-generated names.
local_response_normalization = gen_nn_ops.lrn

# pylint: disable=protected-access


class _NonAtrousConvolution(object):
  """Helper class for _non_atrous_convolution.
  Note that this class assumes that shapes of input and filter passed to
  __call__ are compatible with input_shape and filter_shape passed to the
  constructor.
  Arguments:
    input_shape: static input shape, i.e. input.get_shape().
    filter_shape: static filter shape, i.e. filter.get_shape().
    padding: see _non_atrous_convolution.
    data_format: see _non_atrous_convolution.
    strides: see _non_atrous_convolution.
    name: see _non_atrous_convolution.
  """

  def __init__(
      self,
      input_shape,
      filter_shape,  # pylint: disable=redefined-builtin
      padding,
      data_format=None,
      strides=None,
      name=None,
      **kwargs):
    filter_shape = filter_shape.with_rank(input_shape.ndims)
    self.padding = padding
    self.name = name
    self.kwargs=kwargs
    input_shape = input_shape.with_rank(filter_shape.ndims)
    if input_shape.ndims is None:
      raise ValueError("Rank of convolution must be known")
    if input_shape.ndims < 3 or input_shape.ndims > 5:
      raise ValueError(
          "`input` and `filter` must have rank at least 3 and at most 5")
    conv_dims = input_shape.ndims - 2
    if strides is None:
      strides = [1] * conv_dims
    elif len(strides) != conv_dims:
      raise ValueError("len(strides)=%d, but should be %d" % (len(strides),
                                                              conv_dims))
    if conv_dims == 1:
      # conv1d uses the 2-d data format names
      if data_format is None or data_format == "NWC":
        data_format_2d = "NHWC"
      elif data_format == "NCW":
        data_format_2d = "NCHW"
      else:
        raise ValueError("data_format must be \"NWC\" or \"NCW\".")
      self.strides = strides[0]
      self.data_format = data_format_2d
      self.conv_op = self._conv1d
    elif conv_dims == 2:
      if data_format is None or data_format == "NHWC":
        data_format = "NHWC"
        strides = [1] + list(strides) + [1]
      elif data_format == "NCHW":
        strides = [1, 1] + list(strides)
      else:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")
      self.strides = strides
      self.data_format = data_format
      self.qconv_op=QConv.q2dconvolution_op
      self.conv_op = gen_nn_ops.conv2d
    elif conv_dims == 3:
      if data_format is None or data_format == "NDHWC":
        strides = [1] + list(strides) + [1]
      elif data_format == "NCDHW":
        strides = [1, 1] + list(strides)
      else:
        raise ValueError("data_format must be \"NDHWC\" or \"NCDHW\". Have: %s"
                         % data_format)
      self.strides = strides
      self.data_format = data_format
      self.conv_op = gen_nn_ops.conv3d
        
  # Note that we need this adapter since argument names for conv1d don't match
  # those for gen_nn_ops.conv2d and gen_nn_ops.conv3d.
  # pylint: disable=redefined-builtin
  def _conv1d(self, input, filter, strides, padding, data_format, name):
    return conv1d(
        value=input,
        filters=filter,
        stride=strides,
        padding=padding,
        data_format=data_format,
        name=name)
  # pylint: enable=redefined-builtin

  def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
    if kwargs['quantizer'] is not None:
        return self.qconv_op(
            input=inp,
            filter=filter,
            quantizer=kwargs['quantizer'],
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format)
    else:
        return self.conv_op(
            input=inp,
            filter=filter,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            name=self.name)


#@tf_export("nn.convolution")
def convolution(
    input,  # pylint: disable=redefined-builtin
    filter,  # pylint: disable=redefined-builtin
    padding,
    strides=None,
    dilation_rate=None,
    name=None,
    data_format=None,
    **kwargs):
  # pylint: disable=line-too-long
  # pylint: enable=line-too-long
  with ops.name_scope(name, "convolution", [input, filter]) as name:
    input = ops.convert_to_tensor(input, name="input")  # pylint: disable=redefined-builtin
    input_shape = input.get_shape()
    filter = ops.convert_to_tensor(filter, name="filter")  # pylint: disable=redefined-builtin
    filter_shape = filter.get_shape()
    op = Convolution(
        input_shape,
        filter_shape,
        padding,
        strides=strides,
        dilation_rate=dilation_rate,
        name=name,
        data_format=data_format,
        **kwargs)
    return op(input, filter)


class Convolution(object):
  """Helper class for convolution.
  Note that this class assumes that shapes of input and filter passed to
  __call__ are compatible with input_shape and filter_shape passed to the
  constructor.
  Arguments
    input_shape: static shape of input. i.e. input.get_shape().
    filter_shape: static shape of the filter. i.e. filter.get_shape().
    padding:  see convolution.
    strides: see convolution.
    dilation_rate: see convolution.
    name: see convolution.
    data_format: see convolution.
  """

  def __init__(self,
               input_shape,
               filter_shape,
               padding,
               strides=None,
               dilation_rate=None,
               name=None,
               data_format=None,
               **kwargs):
    """Helper function for convolution."""
    num_total_dims = filter_shape.ndims
    if num_total_dims is None:
      num_total_dims = input_shape.ndims
    if num_total_dims is None:
      raise ValueError("rank of input or filter must be known")

    num_spatial_dims = num_total_dims - 2

    try:
      input_shape.with_rank(num_spatial_dims + 2)
    except ValueError:
      ValueError("input tensor must have rank %d" % (num_spatial_dims + 2))

    try:
      filter_shape.with_rank(num_spatial_dims + 2)
    except ValueError:
      ValueError("filter tensor must have rank %d" % (num_spatial_dims + 2))

    if data_format is None or not data_format.startswith("NC"):
      input_channels_dim = input_shape[num_spatial_dims + 1]
      spatial_dims = range(1, num_spatial_dims + 1)
    else:
      input_channels_dim = input_shape[1]
      spatial_dims = range(2, num_spatial_dims + 2)

    if not input_channels_dim.is_compatible_with(
        filter_shape[num_spatial_dims]):
      raise ValueError(
          "number of input channels does not match corresponding dimension of "
          "filter, {} != {}".format(input_channels_dim,
                                    filter_shape[num_spatial_dims]))

    strides, dilation_rate = nn_ops._get_strides_and_dilation_rate(
        num_spatial_dims, strides, dilation_rate)

    self.input_shape = input_shape
    self.filter_shape = filter_shape
    self.data_format = data_format
    self.strides = strides
    self.name = name
    self.conv_op = _WithSpaceToBatch(
        input_shape,
        dilation_rate=dilation_rate,
        padding=padding,
        build_op=self._build_op,
        filter_shape=filter_shape,
        spatial_dims=spatial_dims,
        data_format=data_format)
    self.kwargs=kwargs

  def _build_op(self, _, padding):
    return _NonAtrousConvolution(
        self.input_shape,
        filter_shape=self.filter_shape,
        padding=padding,
        data_format=self.data_format,
        strides=self.strides,
        name=self.name,
        **self.kwargs)

  def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
    return self.conv_op(inp, filter)


#@tf_export("nn.atrous_conv2d")
def atrous_conv2d(value, filters, rate, padding, name=None, **kwargs):
  return convolution(
      input=value,
      filter=filters,
      padding=padding,
      dilation_rate=np.broadcast_to(rate, (2,)),
      name=name,
      **kwargs)


#@tf_export("nn.conv2d_transpose")
def conv2d_transpose(
    value,
    filter,  # pylint: disable=redefined-builtin
    output_shape,
    strides,
    padding="SAME",
    data_format="NHWC",
    name=None):
  with ops.name_scope(name, "conv2d_transpose",
                      [value, filter, output_shape]) as name:
    if data_format not in ("NCHW", "NHWC"):
      raise ValueError("data_format has to be either NCHW or NHWC.")
    value = ops.convert_to_tensor(value, name="value")
    filter = ops.convert_to_tensor(filter, name="filter")  # pylint: disable=redefined-builtin
    axis = 3 if data_format == "NHWC" else 1
    if not value.get_shape()[axis].is_compatible_with(filter.get_shape()[3]):
      raise ValueError("input channels does not match filter's input channels, "
                       "{} != {}".format(value.get_shape()[axis],
                                         filter.get_shape()[3]))

    output_shape_ = ops.convert_to_tensor(output_shape, name="output_shape")
    if not output_shape_.get_shape().is_compatible_with(tensor_shape.vector(4)):
      raise ValueError("output_shape must have shape (4,), got {}".format(
          output_shape_.get_shape()))

    if isinstance(output_shape, (list, np.ndarray)):
      # output_shape's shape should be == [4] if reached this point.
      if not filter.get_shape()[2].is_compatible_with(output_shape[axis]):
        raise ValueError(
            "output_shape does not match filter's output channels, "
            "{} != {}".format(output_shape[axis],
                              filter.get_shape()[2]))

    if padding != "VALID" and padding != "SAME":
      raise ValueError("padding must be either VALID or SAME:"
                       " {}".format(padding))

    return gen_nn_ops.conv2d_backprop_input(
        input_sizes=output_shape_,
        filter=filter,
        out_backprop=value,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name)
