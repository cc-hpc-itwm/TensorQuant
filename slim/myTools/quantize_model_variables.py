"""
File which loads a model, quantizes all variables and stores them back.


Author: Dominik Loroch
"""

import tensorflow as tf
import sys, os

from Quantize import FixedPoint as fp

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'quantization_method', 'fixed_point',
    'Used quantization method.')

FLAGS = tf.app.flags.FLAGS


quantization_fn_map = {'fixed_point': fp.fixTensor
               }

if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
else:
      checkpoint_path = FLAGS.checkpoint_path

quantization_fn = quantization_fn_map[FLAGS.quantization_method]

with tf.Session() as sess:
    print('Converting %s'%(checkpoint_path))
    saver = tf.train.import_meta_graph(checkpoint_path+'.meta')
    saver.restore(sess, checkpoint_path)
    print('Restorable variables:')
    for var in slim.get_variables_to_restore():
        print('> %s'%(var.name))
        quantization_fn(var,sess,8,4)

    new_checkpoint_path = FLAGS.checkpoint_path+'-'+FLAGS.quantization_method
    if not os.path.exists(new_checkpoint_path+'/'):
        os.makedirs(new_checkpoint_path+'/')
    print('Saving %s'%(new_checkpoint_path))
    saver = tf.train.Saver()
    saver.save(sess,new_checkpoint_path+'/model')
