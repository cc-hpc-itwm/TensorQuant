# File copied from tensorflow/python/training/gradient_descent.py
# Minor modifications are applied to enable intrinsic quantization

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class GradientDescentOptimizer(optimizer.Optimizer):
  """Optimizer that implements the quantized gradient descent algorithm.
  """

  def __init__(self, learning_rate, 
                    use_locking=False, 
                    name="Quantized_GradientDescent",
                    quantizer=None):
    """Construct a new gradient descent optimizer.
    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".
      quantizer: Intrinsic quantizer, applied to new value for variable before it is assigned.
    """
    super(GradientDescentOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self.quantizer = quantizer

  def _apply_dense(self, grad, var):
    if self.quantizer is None:
        return training_ops.apply_gradient_descent(
            var,
            math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            grad,
            use_locking=self._use_locking).op
    else: 
        lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        '''
        delta = self.quantizer.quantize(
                self.quantizer.quantize(grad)*self.quantizer.quantize(lr) )
        new_var = self.quantizer.quantize(var - delta)
        '''
        delta = grad*lr
        new_var = self.quantizer.quantize(var - delta)
        return var.assign(new_var).op
  

  def _resource_apply_dense(self, grad, handle):
    raise IOError('not implemented for quantized SGD!')
    '''
    return training_ops.resource_apply_gradient_descent(
        handle.handle, math_ops.cast(self._learning_rate_tensor,
                                     grad.dtype.base_dtype),
        grad, use_locking=self._use_locking)
    '''

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    raise IOError('not implemented for quantized SGD!')
    '''
    return resource_variable_ops.resource_scatter_add(
        handle.handle, indices, -grad * self._learning_rate)
    '''

  def _apply_sparse_duplicate_indices(self, grad, var):
    raise IOError('not implemented for quantized SGD!')
    '''
    delta = ops.IndexedSlices(
        grad.values *
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.indices, grad.dense_shape)
    return var.scatter_sub(delta, use_locking=self._use_locking)
    '''

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
