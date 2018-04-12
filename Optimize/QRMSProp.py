# File copied from tensorflow/python/training/rmsprop.py
# Minor modifications are applied to enable intrinsic quantization

"""One-line documentation for rmsprop module.
rmsprop algorithm [tieleman2012rmsprop]
A detailed description of rmsprop.
- maintain a moving (discounted) average of the square of gradients
- divide gradient by the root of this average
mean_square = decay * mean_square{t-1} + (1-decay) * gradient ** 2
mom = momentum * mom{t-1} + learning_rate * g_t / sqrt(mean_square + epsilon)
delta = - mom
The centered version additionally maintains a moving (discounted) average of the
gradients, and uses that average to estimate the variance:
mean_grad = decay * mean_square{t-1} + (1-decay) * gradient
mean_square = decay * mean_square{t-1} + (1-decay) * gradient ** 2
mom = momentum * mom{t-1} + learning_rate * g_t /
    sqrt(mean_square - mean_grad**2 + epsilon)
delta = - mom
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

import tensorflow as tf

class RMSPropOptimizer(optimizer.Optimizer):
  """Optimizer that implements the RMSProp algorithm.
  See the [paper](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
  """

  def __init__(self,
               learning_rate,
               decay=0.9,
               momentum=0.0,
               epsilon=1e-10,
               use_locking=False,
               centered=False,
               name="RMSProp",
               quantizer=None):
    """Construct a new RMSProp optimizer.
    Note that in dense implement of this algorithm, m_t and v_t will
    update even if g is zero, but in sparse implement, m_t and v_t
    will not update in iterations g is zero.
    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      decay: Discounting factor for the history/coming gradient
      momentum: A scalar tensor.
      epsilon: Small value to avoid zero denominator.
      use_locking: If True use locks for update operation.
      centered: If True, gradients are normalized by the estimated variance of
        the gradient; if False, by the uncentered second moment. Setting this to
        True may help with training, but is slightly more expensive in terms of
        computation and memory. Defaults to False.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "RMSProp".
    """
    super(RMSPropOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._decay = decay
    self._momentum = momentum
    self._epsilon = epsilon
    self._centered = centered
    self.quantizer=quantizer

    # Tensors for learning rate and momentum.  Created in _prepare.
    self._learning_rate_tensor = None
    self._decay_tensor = None
    self._momentum_tensor = None
    self._epsilon_tensor = None
    

  def _create_slots(self, var_list):
    for v in var_list:
      val_rms = constant_op.constant(1.0, dtype=v.dtype, shape=v.get_shape())
      self._get_or_make_slot(v, val_rms, "rms", self._name)
      if self._centered:
        val_mg = constant_op.constant(0.0, dtype=v.dtype, shape=v.get_shape())
        self._get_or_make_slot(v, val_mg, "mg", self._name)
      self._zeros_slot(v, "momentum", self._name)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                    name="learning_rate")
    self._decay_tensor = ops.convert_to_tensor(self._decay, name="decay")
    self._momentum_tensor = ops.convert_to_tensor(self._momentum,
                                        name="momentum")
    self._epsilon_tensor = ops.convert_to_tensor(self._epsilon,
                                        name="epsilon")
    if self.quantizer is not None:
        self._learning_rate_tensor = self.quantizer.quantize(self._learning_rate_tensor)
        self._decay_tensor = self.quantizer.quantize(self._decay_tensor)
        self._momentum_tensor = self.quantizer.quantize(self._momentum_tensor)
        self._epsilon_tensor = self.quantizer.quantize(self._epsilon_tensor)

  def _apply_rmsprop(self,var,
              rms,
              mom,
              learning_rate_tensor,
              decay_tensor,
              momentum_tensor,
              epsilon_tensor,
              grad,
              quantizer):
        grad=quantizer.quantize(grad)
        rms = rms.assign(quantizer.quantize(
                rms + quantizer.quantize(
                quantizer.quantize(quantizer.quantize(tf.square(grad)) - rms) * 
                quantizer.quantize(1 - decay_tensor) )
              ))
        mom = mom.assign(quantizer.quantize(
                quantizer.quantize(mom * momentum_tensor) + 
                quantizer.quantize(
                    quantizer.quantize(grad * learning_rate_tensor) / 
                    quantizer.quantize(tf.sqrt(rms + epsilon_tensor))
                )
              ))
        var = var.assign(quantizer.quantize(var  - mom))
        return var

  def _apply_centered_rmsprop(self,var,
              mg,
              rms,
              mom,
              learning_rate_tensor,
              decay_tensor,
              momentum_tensor,
              epsilon_tensor,
              grad,
              quantizer):
        grad=quantizer.quantize(grad)
        rms = rms.assign(quantizer.quantize(
                rms + quantizer.quantize(
                quantizer.quantize(quantizer.quantize(tf.square(grad)) - rms) * 
                quantizer.quantize(1 - decay_tensor) )
              ))
        mg = mg.assign(quantizer.quantize(
                mg + 
                quantizer.quantize(
                    quantizer.quantize(grad - mg) * 
                    quantizer.quantize(1 - decay_tensor) 
                )
            ))
        denom = quantizer.quantize(rms - quantizer.quantize(tf.square(mg))) + epsilon_tensor;
        mom = mom.assign(quantizer.quantize(
                    quantizer.quantize(mom * momentum_tensor) +
                    quantizer.quantize( 
                        quantizer.quantize(grad * learning_rate_tensor) / 
                        quantizer.quantize(tf.sqrt(denom))
                    )
                ))
        var = var.assign(quantizer.quantize(var - mom))
        return var

  def _apply_dense(self, grad, var):
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    if self._centered:
      mg = self.get_slot(var, "mg")
      if self.quantizer is None:
        return training_ops.apply_centered_rms_prop(
          var,
          mg,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad,
          use_locking=self._use_locking).op
      else:
        return self._apply_centered_rmsprop(
          var,
          mg,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad,
          self.quantizer).op
    else:
      if self.quantizer is None:
        return training_ops.apply_rms_prop(
          var,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad,
          use_locking=self._use_locking).op
      else:
        return self._apply_rmsprop(
          var,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad,
          self.quantizer).op

  def _resource_apply_dense(self, grad, var):
    raise IOError('not implemented for quantized RMSProp!')
    '''
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    if self._centered:
      mg = self.get_slot(var, "mg")
      return training_ops.resource_apply_centered_rms_prop(
          var.handle,
          mg.handle,
          rms.handle,
          mom.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, grad.dtype.base_dtype),
          grad,
          use_locking=self._use_locking)
    else:
      return training_ops.resource_apply_rms_prop(
          var.handle,
          rms.handle,
          mom.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, grad.dtype.base_dtype),
          grad,
          use_locking=self._use_locking)
    '''

  def _apply_sparse(self, grad, var):
    raise IOError('not implemented for quantized RMSProp!')
    '''
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    if self._centered:
      mg = self.get_slot(var, "mg")
      return training_ops.sparse_apply_centered_rms_prop(
          var,
          mg,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad.values,
          grad.indices,
          use_locking=self._use_locking)
    else:
      return training_ops.sparse_apply_rms_prop(
          var,
          rms,
          mom,
          math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
          math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
          math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
          grad.values,
          grad.indices,
          use_locking=self._use_locking)
    '''

  def _resource_apply_sparse(self, grad, var, indices):
    raise IOError('not implemented for quantized RMSProp!')
    '''
    rms = self.get_slot(var, "rms")
    mom = self.get_slot(var, "momentum")
    if self._centered:
      mg = self.get_slot(var, "mg")
      return training_ops.resource_sparse_apply_centered_rms_prop(
          var.handle,
          mg.handle,
          rms.handle,
          mom.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          math_ops.cast(self._decay_tensor, grad.dtype),
          math_ops.cast(self._momentum_tensor, grad.dtype),
          math_ops.cast(self._epsilon_tensor, grad.dtype),
          grad,
          indices,
          use_locking=self._use_locking)
    else:
      return training_ops.resource_sparse_apply_rms_prop(
          var.handle,
          rms.handle,
          mom.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          math_ops.cast(self._decay_tensor, grad.dtype),
          math_ops.cast(self._momentum_tensor, grad.dtype),
          math_ops.cast(self._epsilon_tensor, grad.dtype),
          grad,
          indices,
          use_locking=self._use_locking)
    '''


    

