# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import utils

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.client import device_lib

#sys.path.append('../Quantize')
from Quantize import QConv
from Quantize import QFullyConnect
from Quantize import QBatchNorm
from Quantize import Factories
from Quantize import Quantizers

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 16,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_string(
    'device', '/gpu:0', 'Device which the model will be placed on.')

tf.app.flags.DEFINE_string(
    'intr_quantizer', '', 'Word width and fractional digits of intrinsic quantizer.'
    'If None, no intrinsic quantizer is applied. Passed as `w,q`.')

tf.app.flags.DEFINE_string(
    'extr_quantizer', '', 'Word width and fractional digits of extrinsic quantizer.'
    'If None, no extrinsic quantizer is applied. Passed as `w,q`.')

tf.app.flags.DEFINE_string(
    'intr_quantize_layers', "", 'layer-types to be quantized intrinsically.'
    'If `None`, no layer is quantized.')

tf.app.flags.DEFINE_string(
    'extr_quantize_layers', "", 'layer-types to be quantized extrinsically.'
    'If `None`, no layer is quantized.')

tf.app.flags.DEFINE_string(
    'output_file', None, 'File in which metrics output is safed.'
    )

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO) #can be WARN or INFO
  #with tf.device(FLAGS.device):
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ########################
    # Determine Quantizers #
    ########################
    intr_quant_width=0
    intr_quant_prec=0
    extr_quant_width=0
    extr_quant_prec=0
    intr_rounding=''
    extr_rounding=''
    
    
    # make a list from the layers to be quantized
    intr_q_layers = FLAGS.intr_quantize_layers.split(",") if FLAGS.intr_quantize_layers !="" else None
    extr_q_layers = FLAGS.extr_quantize_layers.split(",") if FLAGS.extr_quantize_layers !="" else None

    tokens = FLAGS.intr_quantizer.split(',')
    if len(tokens)>=2:
        intr_quant_width = int(tokens[0])
        intr_quant_prec = int(tokens[1])
    if len(tokens)==3:
        intr_rounding = tokens[2]

    if FLAGS.intr_quantizer !='' and intr_q_layers is not None:
        #tokens = FLAGS.intr_quantizer.split(',')
        #intr_quant_width = int(tokens[0])
        #intr_quant_prec = int(tokens[1])
        #intr_rounding = tokens[2]
        if intr_quant_width > intr_quant_prec and intr_quant_prec >= 0:
                intr_q_map = dict.fromkeys( intr_q_layers,
                                utils.quantizer_selector(intr_rounding, 
                                    quant_width=intr_quant_width, quant_prec=intr_quant_prec) )
        else:
            raise ValueError('Intrinsic Quantizer initialized with invalid values: (%d,%d)'
                                %(intr_quant_width,intr_quant_prec))
    else:
        intr_q_map=None

    # mapping for the extrinsic quantizer
    if FLAGS.extr_quantizer !='' and extr_q_layers is not None:
        tokens = FLAGS.extr_quantizer.split(',')
        extr_quant_width = int(tokens[0])
        extr_quant_prec = int(tokens[1])
        extr_rounding = tokens[2]
        if extr_quant_width > extr_quant_prec and extr_quant_prec >= 0:
                extr_q_map = dict.fromkeys( extr_q_layers,
                                utils.quantizer_selector(extr_rounding, 
                                    quant_width=extr_quant_width, quant_prec=extr_quant_prec) )
        
        else:
            raise ValueError('Intrinsic Quantizer initialized with invalid values: (%d,%d)'
                                %(extr_quant_width,extr_quant_prec))
    else:
        extr_q_map=None

    ####################
    # Select the model #
    ####################
    if 'resnet' in FLAGS.model_name:
        labels_offset=1
    else:
        labels_offset=0
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - labels_offset),
        is_training=False,
        intr_q_map=intr_q_map, extr_q_map=extr_q_map)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=8 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size*4)
    [image, label] = provider.get(['image', 'label'])
    label -= labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)
    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    ####################
    # Define the model #
    ####################
    start_time_build = time.time()

    # on single GPU:   
    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    
    logits, endpoints = network_fn(images)
    predictions= tf.argmax(logits, 1)
    labels = tf.squeeze(labels)
    used_gpus=1
    
    #for var in endpoints:
    #    print(var)

    '''
    # on all GPUs:
    available_gpus=get_available_gpus() 
    images={}
    labels={}
    for dev in available_gpus:
      images[dev], labels[dev] = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=10 * FLAGS.batch_size)
    logits={}
    endpoints={}
    predictions={}
    first=True
    for dev in available_gpus:
        with tf.device(dev):
            if first:
                logits[dev], endpoints[dev] = network_fn(images[dev],reuse=None)
                first=False
            else:
                logits[dev], endpoints[dev] = network_fn(images[dev],reuse=True)
        predictions[dev]= tf.argmax(logits[dev], 1)
        labels[dev] = tf.squeeze(labels[dev])
    predictions = tf.stack(list(predictions.values()),axis=0)
    labels = tf.stack(list(labels.values()),axis=0)
    used_gpus=len(available_gpus)
    print('Using %d GPUs.'%(used_gpus))
    '''
    
    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        #'Recall_5': slim.metrics.streaming_recall_at_k(logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1    
    if FLAGS.max_num_batches and FLAGS.max_num_batches > 0:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      #num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
      num_batches = math.ceil(dataset.num_samples / (float(FLAGS.batch_size)*used_gpus) )

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path
    if checkpoint_path is None:
        raise ValueError('No Checkpoint found!')
    tf.logging.info('Evaluating %s' % checkpoint_path)

    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    #config.gpu_options.allocator_type = 'BFC'
    # Run Session
    print('Running %s for %d iterations'%(FLAGS.model_name,num_batches))
    # print('placed on %s'%(dev_name))
    print('layers: %s (intr), %s (extr)'%(FLAGS.intr_quantize_layers,FLAGS.extr_quantize_layers))
    print('intr:%s, extr:%s'%(FLAGS.intr_quantizer,FLAGS.extr_quantizer))

    start_time_simu = time.time()
    metric_values = slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()), 
        final_op=list(names_to_values.values()),
        variables_to_restore=variables_to_restore,
        session_config=config)
    runtime=time.time()-start_time_simu
    buildtime=start_time_simu-start_time_build
    print('Buildtime: %f sec'%buildtime)
    print('Runtime: %f sec'%runtime)

    # tf.train.export_meta_graph(filename=FLAGS.checkpoint_path+'/model.meta')

    # write data to file in json
    if FLAGS.output_file is not None:
      print('Writing results to file %s'%(FLAGS.output_file))
      with open(FLAGS.output_file,'a') as hfile:
        hfile.write( "{\n")
        hfile.write( '  "accuracy":%f,\n'%(metric_values[0]) )
        hfile.write( '  "net":"%s",\n'%(FLAGS.model_name) )
        hfile.write( '  "samples":%d,\n'%(num_batches*FLAGS.batch_size*used_gpus) )
        hfile.write( '  "intr_q_w":%d,\n'%(intr_quant_width) )
        hfile.write( '  "intr_q_f":%d,\n'%(intr_quant_prec) )
        hfile.write( '  "intr_layers":"%s",\n'%(FLAGS.intr_quantize_layers) )
        hfile.write( '  "intr_rounding":"%s",\n'%(intr_rounding) )
        hfile.write( '  "extr_q_w":%d,\n'%(extr_quant_width) )
        hfile.write( '  "extr_q_f":%d,\n'%(extr_quant_prec) )
        hfile.write( '  "extr_layers":"%s",\n'%(FLAGS.extr_quantize_layers) )
        hfile.write( '  "extr_rounding":"%s",\n'%(extr_rounding) )
        hfile.write( '  "runtime":%f\n'%(runtime) )
        hfile.write( "}\n")
    # Plot data

if __name__ == '__main__':
  tf.app.run()
