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


NAME='sparse_grad'
PATH="./alexnet-model/sparse_extrWeight_thresh_0.1/"

var_name='total_loss'
#keys = ['total_loss', ]

FILES=[]
filelist=os.listdir(PATH)
filelist=sorted(filelist)
for f in filelist:
    if '.tfevents' in f:
        FILES.append(f)

steps=[]
loss=[]
for f in FILES:
        print('Opening file %s'%PATH+f)
        try:
            for e in tf.train.summary_iterator(PATH+'/'+f):
              for v in e.summary.value:
                  #print('%s'%(v.tag))
                  if 'acc' in v.tag:
                    print('%s'%(v.tag))
                  if var_name in v.tag:
                      steps.append(e.step)
                      loss.append(v.simple_value)
        except Exception as e:
            print(str(e))
data=[steps,loss]

with open(PATH+NAME+'.json','w') as outfile:
    json.dump(data,outfile)

