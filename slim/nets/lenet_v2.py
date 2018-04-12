# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Contains a variant of the LeNet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from Quantize import QConv
from Quantize import QFullyConnect

slim = tf.contrib.slim

####################
### Quantizer
#intr_quantizer = None # Quantizers.FixedPointQuantizer(8,4)
#extr_quantizer = None # Quantizers.NoQuantizer()
#quantizer = None
####################


#conv2d = Factories.conv2d_factory(intr_quantizer=intr_quantizer, extr_quantizer=extr_quantizer)
#fully_connected = Factories.fully_connected_factory(intr_quantizer=intr_quantizer, extr_quantizer=extr_quantizer)



def lenet(images, num_classes=10, is_training=False, reuse=None,
          dropout_keep_prob=0.5,
          prediction_fn=slim.softmax,
          scope='LeNet',
          **kwargs):
  """Creates a variant of the LeNet model.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:

        logits = lenet.lenet(images, is_training=False)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)

  Args:
    images: A batch of `Tensors` of size [batch_size, height, width, channels].
    num_classes: the number of classes in the dataset.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  conv2d=kwargs["conv2d"]
  max_pool2d=kwargs["max_pool2d"]
  fully_connected = kwargs["fully_connected"]

  def one_way(in_net, filters, size, padding='SAME'):
    net = conv2d(in_net, filters, [size, 1], padding=padding, scope='conv_%dx1_1'%size)
    net = conv2d(net, filters, [1, size], padding=padding, scope='conv_1x%d_2'%size)
    return net

  def two_way(in_net, filters, size):
    subnet1 = conv2d(in_net, filters, [5, 1], padding='VALID', scope='conv_%dx1'%size)
    subnet1 = tf.transpose(subnet1, perm=[0,3,1,2])
    subnet2 = conv2d(in_net, filters, [1, 5], padding='VALID', scope='conv_1x%d'%size)
    subnet2 = tf.transpose(subnet2, perm=[0,3,1,2])
    net = tf.matmul(subnet1,subnet2)
    net = tf.transpose(net, perm=[0,2,3,1])
    return net

  end_points = {}

  with tf.variable_scope(scope, 'LeNet', [images, num_classes],reuse=reuse):
    net = images
    
    end_point = 'Layer01'
    with tf.variable_scope(end_point):
        #net = conv2d(images, 32, [5, 5], scope='conv1')
        net = one_way(net, 32, 5)
        net = max_pool2d(net, [2, 2], 2, scope='pool')
    end_points[end_point] = net

    end_point = 'Layer02'
    with tf.variable_scope(end_point):
        #net = conv2d(net, 64, [5, 5], scope='conv2')
        net = one_way(net, 64, 5)
        net = max_pool2d(net, [2, 2], 2, scope='pool1')
    end_points[end_point] = net

    end_point = 'Layer03'
    with tf.variable_scope(end_point):
        #net = slim.flatten(net)
        #net = fully_connected(net, 1024, scope='fc3')
        net = one_way(net, 1024, 7, padding='VALID')
        net = slim.flatten(net)
        net = slim.dropout(net, dropout_keep_prob, 
                    is_training=is_training,scope='dropout3')
    end_points[end_point] = net

    end_point = 'Layer04'
    with tf.variable_scope(end_point):
        logits = fully_connected(net, num_classes, activation_fn=None, scope='fc4')
        #net = one_way(net, num_classes, 1, conv2d, max_pool2d)
        #logits = slim.flatten(net)
  end_points['Logits'] = logits
  end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points
lenet.default_image_size = 28


def lenet_arg_scope(weight_decay=0.0):
  """Defines the default lenet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected,
       QConv.conv2d, QFullyConnect.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
      activation_fn=tf.nn.relu) as sc:
    return sc
