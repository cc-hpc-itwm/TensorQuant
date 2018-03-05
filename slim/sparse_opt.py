import tensorflow as tf
import utils
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
    'model_name', 'inception_v1', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'layers_file', '', 'Location of file containing all the layer IDs.'
    'If empty, no optimization.')

tf.app.flags.DEFINE_string(
    'tmp_qmap', 'tmp_qmap.json', 'Location of temporarily generated quantizer map.'
    'This file does not have to exist, it will be generated and it is temporary.')

tf.app.flags.DEFINE_string(
    'intr_qmap', '', 'Location of intrinsic quantizer map.'
    '')

tf.app.flags.DEFINE_string(
    'extr_qmap', '', 'Location of extrinsic quantizer map.'
    '')

tf.app.flags.DEFINE_string(
    'weight_qmap', '', 'Location of weight quantizer map.'
    '')

tf.app.flags.DEFINE_string(
    'data_file', 'results.json', 'Location of file with last results.'
    '')

tf.app.flags.DEFINE_string(
    'optimizer_init', 'sparse,1', 'Starting value for optimizer.'
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
INTR_QMAP=      FLAGS.intr_qmap # location of intrinsic quantizer map
EXTR_QMAP=      FLAGS.extr_qmap # location of extrinsic quantizer map
WEIGHT_QMAP=    FLAGS.weight_qmap # location of weight quantizer map

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
BREAK_CONDITION = 0.0001 # if thresh below this condition

def run_baseline():
    eval_execution_str="python eval_image_classifier.py "
    # restoring and logging
    eval_execution_str+="--checkpoint_path=%s "%TRAIN_DIR
    eval_execution_str+="--eval_dir=%s "%EVAL_DIR
    # dataset
    eval_execution_str+="--dataset_dir=%s "%DATASET_DIR
    eval_execution_str+="--dataset_name=%s "%DATASET_NAME
    eval_execution_str+="--dataset_split_name=%s "%DATASET_SPLIT_NAME
    # model and batchsize
    eval_execution_str+="--model_name=%s "%MODEL_NAME
    eval_execution_str+="--max_num_batches=%d "%MAX_NUM_BATCHES
    eval_execution_str+="--batch_size=%d "%BATCH_SIZE
    eval_execution_str+="--output_file=%s "%DATA_FILE
    eval_execution_str+="--comment=\"%s\" "%("type=baseline")
    os.system(eval_execution_str) # call evaluation script


def run_evaluation(layer, qtype, thresh):
    #generate quantizer map for evaluation
    qmap={layer:"%s,%f"%(qtype, thresh)}
    with open(TMP_QMAP,'w') as hfile:
        json.dump(qmap, hfile)
    _type = 'extr'

    # evaluate current setting
    eval_execution_str="python eval_image_classifier.py "
    # restoring and logging
    eval_execution_str+="--checkpoint_path=%s "%TRAIN_DIR
    eval_execution_str+="--eval_dir=%s "%EVAL_DIR
    # dataset
    eval_execution_str+="--dataset_dir=%s "%DATASET_DIR
    eval_execution_str+="--dataset_name=%s "%DATASET_NAME
    eval_execution_str+="--dataset_split_name=%s "%DATASET_SPLIT_NAME
    # model and batchsize
    eval_execution_str+="--model_name=%s "%MODEL_NAME
    eval_execution_str+="--max_num_batches=%d "%MAX_NUM_BATCHES
    eval_execution_str+="--batch_size=%d "%BATCH_SIZE
    # evaluation and quantization
    eval_execution_str+="--output_file=%s "%DATA_FILE
    comment= "type=%s, layer=%s, thresh=%f"%(_type, layer, thresh)
    eval_execution_str+="--comment=\"%s\" "%comment
    eval_execution_str+="--extr_qmap=%s "%TMP_QMAP # use the same qmap
    #eval_execution_str+="--weight_qmap=%s "%TMP_QMAP
    os.system(eval_execution_str) # call evaluation script


def main(_):
    print("Optimizing %s."%(MODEL_NAME))

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
    qtype, qargs = utils.split_quantizer_str(OPTIMIZER_INIT)
    # preprocess quantizer settings
    for key in qmap_template:
        qmap_template[key] = {"type":qtype, "thresh":float(qargs[0]), 
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
                            qmap_template[key]['thresh'])


        # get and process data
        data=[]
        with open(DATA_FILE,'r') as hfile:
            data = json.load(hfile)

        layers= utils.get_vals_from_comments('layer', WORD_REGEX, data)
        utils.get_vals_from_comments('thresh', NUMBER_REGEX, data)

        all_optimal=True #assume everything is optimal in this run
        for dat in data:
            layer = dat['layer']
            # check if current data is baseline
            if 'baseline' in dat['comment']:
                continue
            # check if current data is from this run or from previous one
            if (dat['thresh'] != qmap_template[layer]['thresh']):
                continue

            rel_acc = dat[METRIC]/baseline_accuracy
            if rel_acc<ACCURACY_MARGIN:
                if dat['thresh']/2 >= BREAK_CONDITION:
                    qmap_template[layer]['thresh'] = dat['thresh']/2
                    all_optimal=False # if one or more layers not optimal, do another run
                    print("Layer %s adjusted to thresh=%f."%(layer,
                                qmap_template[layer]['thresh']))
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
        qmap[key] = "%s,%s"%(qmap_template[key]['type'],
                qmap_template[key]['thresh'])
    opt_file = TRAIN_DIR+'/optimal_sparse.json' 
    with open(opt_file,'w') as hfile:
            json.dump(qmap, hfile)
    print("Optimal setup written to %s."%(opt_file))

if __name__ == '__main__':
  tf.app.run()
