"""A network comprising of fully connected layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# imports quantized versions of layers, used in slim.arg_scope
from Quantize import *

slim = tf.contrib.slim


def step_fn(inp):
    return 0.5*(tf.sign(inp)+1)


def fcnet(images, num_classes=10, is_training=False, reuse=None,
          prediction_fn=slim.softmax,
          scope='FCNet',
          **kwargs):

  end_points = {}

  fully_connected=kwargs['fully_connected']

  with tf.variable_scope(scope, 'FCNet', [images, num_classes],reuse=reuse):
        net = slim.flatten(images)
        net = step_fn(net) # process black and white images
        #net = fully_connected(net, 1024, scope='fc1')
        logits = fully_connected(net, num_classes, scope='logits')
  end_points['Logits'] = logits
  end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points

fcnet.default_image_size = 28

def fcnet_arg_scope(weight_decay=0.0):
  with slim.arg_scope(
      [slim.fully_connected,
       QFullyConnect.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
      #activation_fn=tf.nn.relu
      #activation_fn=step_fn
      #activation_fn=tf.tanh
      activation_fn=tf.sign
      ) as sc:
    return sc
