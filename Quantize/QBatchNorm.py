# Copied from tensorflow/contrib/layers/python/layers/layers.py
# Minor modifications are applied to enable intrinsic quantization

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import functools
import six

from tensorflow.contrib.layers.python.layers import layers as slim_layers
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
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
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages

from Quantize import Quantizers

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DATA_FORMAT_NCDHW = 'NCDHW'
DATA_FORMAT_NDHWC = 'NDHWC'

# tensorflow/contrib/layers/python/layers/layers.py
@add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               activation_fn=None,
               param_initializers=None,
               param_regularizers=None,
               updates_collections=ops.GraphKeys.UPDATE_OPS,
               is_training=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               batch_weights=None,
               fused=False,
               data_format=DATA_FORMAT_NHWC,
               zero_debias_moving_mean=False,
               scope=None,
               renorm=False,
               renorm_clipping=None,
               renorm_decay=0.99,
               quantizer=None,
               use_quantized_weights=True):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.
    "Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift"
    Sergey Ioffe, Christian Szegedy
  Can be used as a normalizer function for conv2d and fully_connected.
  Note: when training, the moving_mean and moving_variance need to be updated.
  By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
  need to be added as a dependency to the `train_op`. For example:
  ```python
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss)
  ```
  One can set updates_collections=None to force the updates in place, but that
  can have a speed penalty, especially in distributed settings.
  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC` and the second dimension if `data_format` is
      `NCHW`.
    decay: Decay for the moving average. Reasonable values for `decay` are close
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
      Lower `decay` value (recommend trying `decay`=0.9) if model experiences
      reasonably good training performance but poor validation and/or test
      performance. Try zero_debias_moving_mean=True for improved stability.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    param_initializers: Optional initializers for beta, gamma, moving mean and
      moving variance.
    param_regularizers: Optional regularizer for beta and gamma.
    updates_collections: Collections to collect the update ops for computation.
      The updates_ops need to be executed with the train_op.
      If None, a control dependency would be added to make sure the updates are
      computed in place.
    is_training: Whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    batch_weights: An optional tensor of shape `[batch_size]`,
      containing a frequency weight for each batch item. If present,
      then the batch normalization uses weighted mean and
      variance. (This can be used to correct for bias in training
      example selection.)
    fused: if `True`, use a faster, fused implementation based on
      nn.fused_batch_norm. If `None`, use the fused implementation if possible.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    zero_debias_moving_mean: Use zero_debias for moving_mean. It creates a new
      pair of variables 'moving_mean/biased' and 'moving_mean/local_step'.
    scope: Optional scope for `variable_scope`.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_decay: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `decay` is still applied
      to get the means and variances for inference.
  Returns:
    A `Tensor` representing the output of the operation.
  Raises:
    ValueError: If `batch_weights` is not None and `fused` is True.
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
  """
  if fused:
    raise ValueError('Quantization is not supported for fused batch norm. ')
    if batch_weights is not None:
      raise ValueError('Weighted mean and variance is not currently '
                       'supported for fused batch norm.')
    if param_regularizers is not None:
      raise ValueError('Regularizers are not currently '
                       'supported for fused batch norm.')
    if renorm:
      raise ValueError('Renorm is not supported for fused batch norm.')

  # Only use _fused_batch_norm (1) if fused is set True or if it is
  # possible to use (currently it doesn't support batch weights,
  # renorm, and the case when rank is neither 2 nor 4),
  # and (2) if used with zero_debias_moving_mean, or an input shape of rank 2,
  # or non-default updates_collections (not implemented in
  # normalization_layers.BatchNormalization yet); otherwise use the fused
  # implementation in normalization_layers.BatchNormalization.
  inputs = ops.convert_to_tensor(inputs)
  rank = inputs.get_shape().ndims
  feature_supported = batch_weights is None and not renorm and rank in [2, 4]
  possible_to_fuse = fused is None and feature_supported
  if (fused or possible_to_fuse) and (
      zero_debias_moving_mean or rank == 2 or
      updates_collections is not ops.GraphKeys.UPDATE_OPS):
    return _fused_batch_norm(
        inputs,
        decay=decay,
        center=center,
        scale=scale,
        epsilon=epsilon,
        activation_fn=activation_fn,
        param_initializers=param_initializers,
        updates_collections=updates_collections,
        is_training=is_training,
        reuse=reuse,
        variables_collections=variables_collections,
        outputs_collections=outputs_collections,
        trainable=trainable,
        data_format=data_format,
        zero_debias_moving_mean=zero_debias_moving_mean,
        scope=scope)

  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')

  layer_variable_getter = slim_layers._build_variable_getter()
  with variable_scope.variable_scope(
      scope, 'BatchNorm', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)

    # Determine whether we can use the core layer class.
    if (batch_weights is None and
        updates_collections is ops.GraphKeys.UPDATE_OPS and
        not zero_debias_moving_mean):
      # Use the core layer class.
      axis = 1 if data_format == DATA_FORMAT_NCHW else -1
      if not param_initializers:
        param_initializers = {}
      beta_initializer = param_initializers.get('beta',
                                                init_ops.zeros_initializer())
      gamma_initializer = param_initializers.get('gamma',
                                                 init_ops.ones_initializer())
      moving_mean_initializer = param_initializers.get(
          'moving_mean', init_ops.zeros_initializer())
      moving_variance_initializer = param_initializers.get(
          'moving_variance', init_ops.ones_initializer())
      if not param_regularizers:
        param_regularizers = {}
      beta_regularizer = param_regularizers.get('beta')
      gamma_regularizer = param_regularizers.get('gamma')
      #This call is mainly used by the slim models
      layer = QBatchNormalization( #normalization_layers.BatchNormalization(
          axis=axis,
          momentum=decay,
          epsilon=epsilon,
          center=center,
          scale=scale,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer,
          moving_mean_initializer=moving_mean_initializer,
          moving_variance_initializer=moving_variance_initializer,
          beta_regularizer=beta_regularizer,
          gamma_regularizer=gamma_regularizer,
          renorm=renorm,
          renorm_clipping=renorm_clipping,
          renorm_momentum=renorm_decay,
          #fused=fused,
          trainable=trainable,
          name=sc.name,
          quantizer=quantizer,
          use_quantized_weights=use_quantized_weights,
          _scope=sc,
          _reuse=reuse
          )
      outputs = layer.apply(inputs, training=is_training)

      # Add variables to collections.
      slim_layers._add_variable_to_collections(
          layer.moving_mean, variables_collections, 'moving_mean')
      slim_layers._add_variable_to_collections(
          layer.moving_variance, variables_collections, 'moving_variance')
      if layer.beta is not None:
        slim_layers._add_variable_to_collections(layer.beta, variables_collections, 'beta')
      if layer.gamma is not None:
        slim_layers._add_variable_to_collections(
            layer.gamma, variables_collections, 'gamma')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return utils.collect_named_outputs(outputs_collections,
                                         sc.original_name_scope, outputs)

    raise ValueError('Only core layer supported for quantized batch norm.')


# tensorflow/python/layers/normalization.py
class QBatchNormalization(normalization_layers.BatchNormalization):
  """Batch Normalization layer from http://arxiv.org/abs/1502.03167.
  "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift"
  Sergey Ioffe, Christian Szegedy
  Arguments:
    axis: Integer, the axis that should be normalized (typically the features
      axis). For instance, after a `Conv2D` layer with
      `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.
    fused: if `True`, use a faster, fused implementation based on
      nn.fused_batch_norm. If `None`, use the fused implementation if possible.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    name: A string, the name of the layer.
  """

  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer=init_ops.zeros_initializer(),
               gamma_initializer=init_ops.ones_initializer(),
               moving_mean_initializer=init_ops.zeros_initializer(),
               moving_variance_initializer=init_ops.ones_initializer(),
               beta_regularizer=None,
               gamma_regularizer=None,
               trainable=True,
               name=None,
               quantizer=None,
               use_quantized_weights=True,
               **kwargs):
    super(QBatchNormalization, self).__init__(
               axis=axis,
               momentum=momentum,
               epsilon=epsilon,
               center=center,
               scale=scale,
               beta_initializer=beta_initializer,
               gamma_initializer=gamma_initializer,
               moving_mean_initializer=moving_mean_initializer,
               moving_variance_initializer=moving_variance_initializer,
               beta_regularizer=beta_regularizer,
               gamma_regularizer=gamma_regularizer,
               trainable=trainable,
               name=name,
               **kwargs)
    self.quantizer=quantizer
    self.use_quantized_weights=use_quantized_weights

  def build(self, input_shape):
        super(QBatchNormalization,self).build(input_shape)
        if self.quantizer is not None and self.use_quantized_weights:
          if self.moving_mean is not None:
            self.moving_mean = self.quantizer.quantize(self.moving_mean)
          if self.moving_variance is not None:
            self.moving_variance = self.quantizer.quantize(self.moving_variance)
          if self.beta is not None:
            self.beta = self.quantizer.quantize(self.beta)
          if self.gamma is not None:
            self.gamma = self.quantizer.quantize(self.gamma)

  # override 'call' to evoke own batch normalization function
  def call(self, inputs, training=False):
    # First, compute the axes along which to reduce the mean / variance,
    # as well as the broadcast shape to be used for all parameters.
    input_shape = inputs.get_shape()
    ndim = len(input_shape)
    reduction_axes = list(range(len(input_shape)))
    del reduction_axes[self.axis]
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[self.axis] = input_shape[self.axis].value

    # Determines whether broadcasting is needed.
    needs_broadcasting = (sorted(reduction_axes) != range(ndim)[:-1])

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = utils.constant_value(training)

    if needs_broadcasting:
      # In this case we must explictly broadcast all parameters.
      if self.center:
        broadcast_beta = array_ops.reshape(self.beta, broadcast_shape)
      else:
        broadcast_beta = None
      if self.scale:
        broadcast_gamma = array_ops.reshape(self.gamma, broadcast_shape)
      else:
        broadcast_gamma = None

    # Determines moments
    if training_value is not False:
      if needs_broadcasting:
        broadcast_mean, broadcast_variance = nn.moments(
            inputs, reduction_axes, keep_dims=True)
        mean = array_ops.reshape(broadcast_mean, [-1])
        variance = array_ops.reshape(broadcast_variance, [-1])
      else:
        mean, variance = nn.moments(inputs, reduction_axes)

      # Prepare updates if necessary.
      if not self.updates:
        mean_update = moving_averages.assign_moving_average(
            self.moving_mean, mean, self.momentum, zero_debias=False)
        variance_update = moving_averages.assign_moving_average(
            self.moving_variance, variance, self.momentum, zero_debias=False)
        # In the future this should be refactored into a self.add_update
        # methods in order to allow for instance-based BN layer sharing
        # across unrelated input streams (e.g. like in Keras).
        self.updates.append(mean_update)
        self.updates.append(variance_update)

    # Normalize batch. We do this inside separate functions for training
    # and inference so as to avoid evaluating both branches.
    def normalize_in_test():
      if needs_broadcasting:
        broadcast_moving_mean = array_ops.reshape(self.moving_mean,
                                                  broadcast_shape)
        broadcast_moving_variance = array_ops.reshape(self.moving_variance,
                                                      broadcast_shape)
      arg_mean = broadcast_moving_mean if needs_broadcasting else self.moving_mean
      arg_variance = broadcast_moving_variance if needs_broadcasting else self.moving_variance 
      arg_beta = broadcast_beta if needs_broadcasting else (self.beta if self.center else None)
      arg_gamma = broadcast_gamma if needs_broadcasting else (self.gamma if self.scale else None)
      if self.quantizer is None:
        return nn.batch_normalization(inputs,
                                  arg_mean,
                                  arg_variance,
                                  arg_beta,
                                  arg_gamma,
                                  self.epsilon)
      else:
        return qbatch_normalization(inputs,
                                  arg_mean,
                                  arg_variance,
                                  arg_beta,
                                  arg_gamma,
                                  self.epsilon,
                                  self.quantizer)

    def normalize_in_training():
      arg_mean = broadcast_mean if needs_broadcasting else mean
      arg_variance = broadcast_variance if needs_broadcasting else variance 
      arg_beta = broadcast_beta if needs_broadcasting else (self.beta if self.center else None)
      arg_gamma = broadcast_gamma if needs_broadcasting else (self.gamma if self.scale else None)
      if self.quantizer is None:
        return nn.batch_normalization(inputs,
                                  arg_mean,
                                  arg_variance,
                                  arg_beta,
                                  arg_gamma,
                                  self.epsilon)
      else:
        return qbatch_normalization(inputs,
                                  arg_mean,
                                  arg_variance,
                                  arg_beta,
                                  arg_gamma,
                                  self.epsilon,
                                  self.quantizer)
    
    
    return utils.smart_cond(training,
                            normalize_in_training,
                            normalize_in_test)


# quantized batch normalization calculation
# tensorflow/python/ops/nn_impl.py
def qbatch_normalization(x,
                        mean,
                        variance,
                        offset,
                        scale,
                        variance_epsilon,
                        quantizer,
                        name=None):
  r"""Batch normalization.
  As described in http://arxiv.org/abs/1502.03167.
  Normalizes a tensor by `mean` and `variance`, and applies (optionally) a
  `scale` \\(\gamma\\) to it, as well as an `offset` \\(\beta\\):
  \\(\frac{\gamma(x-\mu)}{\sigma}+\beta\\)
  `mean`, `variance`, `offset` and `scale` are all expected to be of one of two
  shapes:
    * In all generality, they can have the same number of dimensions as the
      input `x`, with identical sizes as `x` for the dimensions that are not
      normalized over (the 'depth' dimension(s)), and dimension 1 for the
      others which are being normalized over.
      `mean` and `variance` in this case would typically be the outputs of
      `tf.nn.moments(..., keep_dims=True)` during training, or running averages
      thereof during inference.
    * In the common case where the 'depth' dimension is the last dimension in
      the input tensor `x`, they may be one dimensional tensors of the same
      size as the 'depth' dimension.
      This is the case for example for the common `[batch, depth]` layout of
      fully-connected layers, and `[batch, height, width, depth]` for
      convolutions.
      `mean` and `variance` in this case would typically be the outputs of
      `tf.nn.moments(..., keep_dims=False)` during training, or running averages
      thereof during inference.
  Args:
    x: Input `Tensor` of arbitrary dimensionality.
    mean: A mean `Tensor`.
    variance: A variance `Tensor`.
    offset: An offset `Tensor`, often denoted \\(\beta\\) in equations, or
      None. If present, will be added to the normalized tensor.
    scale: A scale `Tensor`, often denoted \\(\gamma\\) in equations, or
      `None`. If present, the scale is applied to the normalized tensor.
    variance_epsilon: A small float number to avoid dividing by 0.
    name: A name for this operation (optional).
  Returns:
    the normalized, scaled, offset tensor.
  """
  with ops.name_scope(name, "batchnorm", [x, mean, variance, scale, offset]):
    #internal calculation of sqrt is NOT quantized! 
    inv = quantizer.quantize( math_ops.sqrt(variance + variance_epsilon) )
    inv = quantizer.quantize( math_ops.reciprocal(inv) )
    if scale is not None:
      inv = quantizer.quantize(inv*scale)
    #return x * inv + (offset - mean * inv if offset is not None else -mean * inv)
    if offset is not None:
        rest = quantizer.quantize( offset - quantizer.quantize(mean * inv) )
    else:
        rest = quantizer.quantize(-mean * inv)
    result = quantizer.quantize( quantizer.quantize(x * inv) + rest )
    return result


