'''
Some Optimizer 01
'''

from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.training.optimizer import Optimizer
import tensorflow as tf


class NM01(Optimizer):
    def __init__(self, learning_rate, quantizer=None, use_locking=False, name='NM01'):
        '''
        Optimizer 01
        '''
        super(NM01, self).__init__(use_locking, name)
        self.lr = learning_rate
        self.quantizer = quantizer

    def _create_slots(self, var_list):
        for v in var_list:
            dynamic_range = constant_op.constant(1, dtype=v.dtype, shape=v.get_shape())
            self._get_or_make_slot(v, dynamic_range, "dynamic_range", self._name)
            self._zeros_slot(v, "grad_counter", self._name)

    def _prepare(self):
        self.lr_tensor = ops.convert_to_tensor(self.lr,
                                        name="learning_rate")

    def _apply_dense(self, grad, var):
        grad_counter = self.get_slot(var, "grad_counter")
        dynamic_range = self.get_slot(var, "dynamic_range")

        #grad_counter_new = grad_counter*tf.cast(tf.sign(grad)==tf.sign(grad_counter), tf.float32)
        grad_counter_new = tf.minimum(tf.maximum(grad_counter+tf.sign(grad),-64),64)
        #grad_counter_new += tf.sign(grad)*tf.cast(grad_counter_new == 0,tf.float32)
        #grad_counter_new = grad_counter+tf.sign(grad)

        _grad=grad
        #grad_thresh=0
        grad_thresh = tf.reduce_mean(tf.abs(_grad*grad_counter_new))        
        #grad_thresh = tf.reduce_max(tf.abs(grad))*0.1
        #_grad = _grad*tf.cast(tf.abs(grad*grad_counter_new)>=grad_thresh,tf.float32)
        #_grad=_grad*dynamic_range
        #_grad=_grad/(tf.reduce_max(_grad))
        #_grad = grad*tf.cast(tf.sign(grad)==tf.sign(grad_counter_new), tf.float32)
        
        #delta = self.quantizer.quantize(self.lr_tensor * tf.abs(grad_counter_new) * grad)
        #delta_max=tf.reduce_max(self.lr_tensor*reduced_grad)  
        #delta_min=tf.reduce_min(self.lr_tensor*reduced_grad)        
        #delta = self.quantizer.quantize(self.lr_tensor * tf.abs(grad_counter_new) * _grad)
        delta = self.lr_tensor * tf.abs(grad_counter_new) * _grad
        #delta = tf.abs(grad_counter_new) * _grad
        #delta = tf.maximum(tf.minimum(delta,delta_max),delta_min)

        var_update = self.quantizer.quantize(var-delta)

        #dynamic_range_update = dynamic_range * (1 + 
        #        tf.cast(tf.reduce_all(tf.abs(var-var_update)<=0.01),tf.float32))
        dynamic_range_update = dynamic_range / (1 + 
                tf.cast(tf.reduce_all(tf.abs(var-var_update)>=0.01),tf.float32))

        #reset grad_counter, if update happened
        grad_counter_update = grad_counter_new * tf.cast( 
                                    (var-var_update) != 0, tf.float32)
        
        var_update_op = state_ops.assign(var, var_update)
        dynamic_range_update_op = state_ops.assign(dynamic_range, dynamic_range_update)
        grad_counter_update_op = state_ops.assign(grad_counter,
                                         grad_counter_update)
        
        return control_flow_ops.group(*[grad_counter_update_op,
                             var_update_op,
                             dynamic_range_update_op])

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

