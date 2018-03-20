"""A network comprising of fully connected layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from Quantize import QFullyConnect

slim = tf.contrib.slim

def fcnet(images, num_classes=10, is_training=False, reuse=None,
          prediction_fn=slim.softmax,
          scope='FCNet',
          fully_connected = slim.fully_connected):

  end_points = {}

  with tf.variable_scope(scope, 'FCNet', [images, num_classes],reuse=reuse):
        #print(images)
        net = slim.flatten(images)
        #print(net)
        net = step_fn(net)
        #net = fully_connected(net, 1024, scope='fc1')
        #end_points['Flatten'] = net
        logits = fully_connected(net, num_classes, scope='logits')
        #print(logits)
  end_points['Logits'] = logits
  end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points

fcnet.default_image_size = 28

def step_fn(inp):
    return 0.5*(tf.sign(inp)+1)

def fcnet_arg_scope(weight_decay=0.0):
  with slim.arg_scope(
      [slim.fully_connected,
       QFullyConnect.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
      #activation_fn=tf.nn.relu
      activation_fn=step_fn
      #activation_fn=tf.sigmoid
      ) as sc:
    return sc
