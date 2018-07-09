from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.training.optimizer import Optimizer
import tensorflow as tf

"""
SGD with LARS.
https://arxiv.org/abs/1708.03888
"""

class Lars(Optimizer):
    def __init__(self, learning_rate, momentum=0.9, weight_decay=0.0005, eta=0.001, use_locking=False, name='Lars', quantizer=None):
        '''
        constructs a new LARS optimizer
        '''
        super(Lars, self).__init__(use_locking, name)
        self.global_lr=learning_rate
        self.mom = momentum
        self.weight_decay=weight_decay
        self.eta=eta
        self.quantizer=quantizer
        

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                momentum = constant_op.constant(0, shape=v.get_shape(), dtype=v.dtype.base_dtype)
            self._get_or_make_slot(v, momentum, "mom", self._name)


    def _apply_dense(self, grad, var):
        momentum = self.get_slot(var, "mom")

        var_d = tf.sqrt(tf.reduce_sum(tf.square(var)))
        grad_d = tf.sqrt(tf.reduce_sum(tf.square(grad)))
        local_lr = var_d / (grad_d + self.weight_decay * var_d)

        momentum_update = self.mom * momentum + self.global_lr * local_lr * (grad+self.weight_decay*var)
        if self.quantizer is not None:
            var_update = self.quantizer.quantize(var-momentum_update)
        else:
            var_update = var-momentum_update
        
        momentum_update_op = state_ops.assign(momentum, momentum_update)
        var_update_op = state_ops.assign(var, var_update)

        return control_flow_ops.group(*[var_update_op, momentum_update_op])

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)


