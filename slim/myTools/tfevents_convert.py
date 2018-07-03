import tensorflow as tf
# from Quantize import *
import os
import matplotlib as mplot
import matplotlib.pyplot as plt
import datetime
import json
import re

def dict_descend(dictionary, key):
    if key not in dictionary:
        dictionary[key]={}
    return dictionary[key]


#NAME='imagenet_sparse_grad_'
NAME='sparse_grad_thresh_'
#NAME='long_train'
#MODEL="cifarnet"
MODEL="resnetv1_14"
#MODEL="lenet"
#MODEL="alexnet"
PATH="./tmp/%s-model/"%(MODEL)
NAMES= [ f for f in os.listdir(PATH) if NAME in f or 'baseline' in f]
FOLDERS= [ PATH+f+'/' for f in NAMES]

var_name='gradient-sparsity'
#keys = ['total_loss', ]





for folder_idx, folder in enumerate(FOLDERS):
    print('Directory %s:'%folder)
    data={}
    FILES=[ folder+f for f in os.listdir(folder) if '.tfevents' in f ]
    #for f in filelist:
    #    if '.tfevents' in f:
    #        FILES.append(f)
    for f in FILES:
        print('    Opening file %s'%f)
        try:
            for e in tf.train.summary_iterator(f):
              data[e.step]={}
              for v in e.summary.value:
                  #print('%s'%(v.tag))
                  #if 'Accuracy' in v.tag:
                  #  print('%s'%(v.tag))
                  if var_name in v.tag:
                      data[e.step][v.tag]=v.simple_value
                  if 'eval/Accuracy' in v.tag:
                      data['accuracy']=v.simple_value
              if not data[e.step]:
                data.pop(e.step)
        except Exception as e:
            print(str(e))

    with open('./experiment_results/'+MODEL+'_'+NAMES[folder_idx]+'.json','w') as outfile:
        json.dump(data,outfile)

