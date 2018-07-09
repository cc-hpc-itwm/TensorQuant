"""
Tries to find a layerwise optimum representation of fixed point quantization.

Run from "slim" directory with:

python fixed_opt.py \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=${DATASET_TEST_NAME} \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${MODEL_NAME} \
    --labels_offset=${LABELS_OFFSET} \
    --preprocessing_name=${PREPROCESSING_NAME} \
    --eval_image_size=${IMG_SIZE} \
    --batch_size=${BATCH_SIZE} \
    --max_num_batches=${NUM_BATCHES} \
    --layers_file=${BASE_DIR}/layers.json \
    --tmp_qmap=${TRAIN_DIR}/tmp_qmap.json \
    --data_file=${EXP_FILE} \
    --optimizer_init="nearest,4,2" \
    --optimizer_mode=${OPTIMIZER_MODE} \
    --margin=1.0 \
    --opt_qmap=${TRAIN_DIR}/opt_${OPTIMIZER_MODE}.json

Author: Dominik Loroch
"""

import tensorflow as tf
import Quantize.utils
from misc import utils
import json
import os

# constants
WORD_REGEX='([\w/]*)'
NUMBER_REGEX='(\d*\.?\d*)'

# Flags
tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'layers_file', '', 'Location of file containing all the layer IDs.'
    'If empty, no optimization.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_string(
    'tmp_qmap', 'tmp_qmap.json', 'Location of temporarily generated quantizer map.'
    'This file does not have to exist, it will be generated and it is temporary.')

tf.app.flags.DEFINE_string(
    'opt_qmap', 'opt_qmap.json', 'The final optimal quantizer map.'
    'Contains the results of the quantizer.')

tf.app.flags.DEFINE_string(
    'data_file', 'opt_results.json', 'Location of file with last results.'
    '')

tf.app.flags.DEFINE_string(
    'optimizer_init', 'nearest,4,2', 'Starting value for optimizer.'
    '')

tf.app.flags.DEFINE_string(
    'optimizer_mode', 'extr', 'quantization of activations ("extr"), weights ("weight"), or both ("extr_weights").'
    '')

tf.app.flags.DEFINE_float(
    'margin', 0.99, 'Margin to achieve.'
    'Optimizer aims to be above this margin.')

FLAGS = tf.app.flags.FLAGS

# evaluation configuration
# pathes to different files and filenames
TRAIN_DIR=      FLAGS.checkpoint_path # checkpoint directory
EVAL_DIR=       "/tmp/tf" # directory to dump summaries into
LAYERS_FILE=    FLAGS.layers_file # available layers
TMP_QMAP=       FLAGS.tmp_qmap # location of temporary quantizer map
OPT_QMAP=       FLAGS.opt_qmap # location of where the final qmap will be saved to

DATASET_DIR= FLAGS.dataset_dir # Where the dataset is saved to.
DATASET_NAME= FLAGS.dataset_name # Name of the dataset, used by dataset factory
DATASET_SPLIT_NAME= FLAGS.dataset_split_name # what subset of the dataset will be used
MODEL_NAME= FLAGS.model_name # topology name used by factory

MAX_NUM_BATCHES= FLAGS.max_num_batches # how many batches will be processed
BATCH_SIZE= FLAGS.batch_size # number of samples per batch

DATA_FILE=FLAGS.data_file # where the evaluation results are written to
utils.remove_file(DATA_FILE) # make sure there is no previous data

METRIC = 'accuracy' # name for metric used in the experiment
ACCURACY_MARGIN = FLAGS.margin # relative accuracy margin to be achieved for each layer

OPTIMIZER_INIT = FLAGS.optimizer_init
BREAK_CONDITION = 24 # if w above this condition

def run_baseline():
    eval_execution_str="python eval_image_classifier.py "
    # restoring and logging
    eval_execution_str+="--checkpoint_path=%s "%TRAIN_DIR
    eval_execution_str+="--eval_dir=%s "%EVAL_DIR
    # dataset
    eval_execution_str+="--dataset_dir=%s "%DATASET_DIR
    eval_execution_str+="--dataset_name=%s "%DATASET_NAME
    eval_execution_str+="--dataset_split_name=%s "%DATASET_SPLIT_NAME
    eval_execution_str+="--labels_offset=%d "%FLAGS.labels_offset
    # model and batchsize
    eval_execution_str+="--model_name=%s "%MODEL_NAME
    eval_execution_str+="--max_num_batches=%d "%MAX_NUM_BATCHES
    eval_execution_str+="--batch_size=%d "%BATCH_SIZE
    if FLAGS.preprocessing_name is not None:
        eval_execution_str+="--preprocessing_name=%s "%FLAGS.preprocessing_name
    if FLAGS.eval_image_size is not None:
        eval_execution_str+="--eval_image_size=%d "%FLAGS.eval_image_size
    # logging
    eval_execution_str+="--output_file=%s "%DATA_FILE
    eval_execution_str+="--comment=\"%s\" "%("type=baseline")
    os.system(eval_execution_str) # call evaluation script


def run_evaluation(layer, qtype, w, q):
    #generate quantizer map for evaluation
    qmap={layer:"%s,%d,%d"%(qtype, w, q)}
    with open(TMP_QMAP,'w') as hfile:
        json.dump(qmap, hfile)
    _type = FLAGS.optimizer_mode

    # evaluate current setting
    eval_execution_str="python eval_image_classifier.py "
    # restoring and logging
    eval_execution_str+="--checkpoint_path=%s "%TRAIN_DIR
    eval_execution_str+="--eval_dir=%s "%EVAL_DIR
    # dataset
    eval_execution_str+="--dataset_dir=%s "%DATASET_DIR
    eval_execution_str+="--dataset_name=%s "%DATASET_NAME
    eval_execution_str+="--dataset_split_name=%s "%DATASET_SPLIT_NAME
    eval_execution_str+="--labels_offset=%d "%FLAGS.labels_offset
    # model and batchsize
    eval_execution_str+="--model_name=%s "%MODEL_NAME
    eval_execution_str+="--max_num_batches=%d "%MAX_NUM_BATCHES
    eval_execution_str+="--batch_size=%d "%BATCH_SIZE
    if FLAGS.preprocessing_name is not None:
        eval_execution_str+="--preprocessing_name=%s "%FLAGS.preprocessing_name
    if FLAGS.eval_image_size is not None:
        eval_execution_str+="--eval_image_size=%d "%FLAGS.eval_image_size
    # evaluation and quantization
    eval_execution_str+="--output_file=%s "%DATA_FILE
    comment= "type=%s, layer=%s, w=%d, q=%d"%(_type,layer, w, q)
    eval_execution_str+="--comment=\"%s\" "%comment
    if "intr" in FLAGS.optimizer_mode:
        eval_execution_str+="--intr_qmap=%s "%TMP_QMAP
    if "extr" in FLAGS.optimizer_mode:
        eval_execution_str+="--extr_qmap=%s "%TMP_QMAP
    if "weight" in FLAGS.optimizer_mode:
        eval_execution_str+="--weight_qmap=%s "%TMP_QMAP
    os.system(eval_execution_str) # call evaluation script


def main(_):
    print("Optimizing %s"%(MODEL_NAME))

    # get baseline
    # ---------------------
    run_baseline()
    with open(DATA_FILE,'r') as hfile:
        baseline_data = json.load(hfile)

    baseline_accuracy=baseline_data[0][METRIC]

    # optimization
    # ---------------------
    # the init file contains all layers to optimize for and initial values
    with open(LAYERS_FILE,'r') as hfile:
        layers = json.load(hfile)
    qmap_template = dict.fromkeys(layers)
    qtype, qargs = Quantize.utils.split_quantizer_str(OPTIMIZER_INIT)
    # preprocess quantizer settings
    for key in qmap_template:
        qmap_template[key] = {"type":qtype, "w":int(qargs[0]), "q":int(qargs[1]), 
                              "optimal":False}

    # start optimization loop
    all_optimal=False
    while not all_optimal:
        # run evaluation for every layer
        for key in qmap_template.keys():
            #skip already optimal layers
            if qmap_template[key]['optimal'] is True:
                continue
            run_evaluation(key, qmap_template[key]['type'],
                            qmap_template[key]['w'],qmap_template[key]['q'])


        # get and process data
        data=[]
        with open(DATA_FILE,'r') as hfile:
            data = json.load(hfile)

        layers= utils.get_vals_from_comments('layer', WORD_REGEX, data)
        utils.get_vals_from_comments('w', NUMBER_REGEX, data)
        utils.get_vals_from_comments('q', NUMBER_REGEX, data)

        all_optimal=True #assume everything is optimal in this run
        for dat in data:
            layer = dat['layer']
            # check if current data is baseline
            if 'baseline' in dat['comment']:
                continue
            # check if current data is from this run or from previous one
            if (dat['w'] != qmap_template[layer]['w'] or
                dat['q'] != qmap_template[layer]['q']):
                continue

            rel_acc = dat[METRIC]/baseline_accuracy
            if rel_acc<ACCURACY_MARGIN:
                if dat['q']+2 < dat['w']:
                    qmap_template[layer]['q'] = dat['q']+2
                    all_optimal=False # if one or more layers not optimal, do another run
                    print("Layer %s adjusted to w=%d, q=%d."%(layer,
                                qmap_template[layer]['w'],qmap_template[layer]['q']))
                elif dat['w']+4 <= BREAK_CONDITION:
                    qmap_template[layer]['w'] = dat['w']+4
                    qmap_template[layer]['q'] = 2
                    all_optimal=False # if one or more layers not optimal, do another run
                    print("Layer %s adjusted to w=%d, q=%d."%(layer,
                                qmap_template[layer]['w'],qmap_template[layer]['q']))
                else:
                    qmap_template[layer]['optimal']=True # skip this layer from optimization
                    print("Break condition reached, layer %s cannot be optimized."%layer)
            else:
                # if margin is already reached, there is no more optimization necessary
                qmap_template[layer]['optimal']=True
                print("Layer %s is optimal."%layer)
        

    # End of optimization loop
    print("Optimization complete.")
    # writing optimal settings to file
    qmap = dict.fromkeys(qmap_template.keys())
    for key in qmap.keys():
        qmap[key] = "%s,%d,%d"%(qmap_template[key]['type'],
                int(qmap_template[key]['w']),int(qmap_template[key]['q']))
    opt_file = OPT_QMAP
    with open(opt_file,'w') as hfile:
            json.dump(qmap, hfile)
    print("Optimal setup written to %s."%(opt_file))

if __name__ == '__main__':
  tf.app.run()
