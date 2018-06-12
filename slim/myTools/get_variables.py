import tensorflow as tf
from TensorQuant.Quantize import Quantizers
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

import DNN_pb2

slim = tf.contrib.slim


MODEL='lenet-model'
PREFIX='LeNet'
PB_NAME=MODEL+".pb"

if tf.gfile.IsDirectory(MODEL):
      checkpoint_path = tf.train.latest_checkpoint(MODEL)
else:
      checkpoint_path = MODEL

export_data={}
with tf.Session() as sess:
    print('Opening %s'%(checkpoint_path))
    saver = tf.train.import_meta_graph(MODEL+'/model.meta')
    saver.restore(sess, checkpoint_path)
    print('Restorable variables:')
    for var in slim.get_variables_to_restore():
        if PREFIX in var.name:
            print('> %s'%(var.name))
            data = np.array(var.eval())
            data = np.transpose(data)
            print(data.shape)
            #=var.eval().tolist()
            export_data[var.name]=data

print("Generating %s"%PB_NAME)
net = DNN_pb2.Net()
net.name='LeNet'
net.num_layers=len(export_data)
for key in export_data.keys():
    layer = net.layers.add()
    layer.name=key
    layer.shape.extend(export_data[key].shape)
    #print(export_data[key].flatten().tolist())
    layer.data.extend(export_data[key].flatten().tolist())
#print(net)
f = open(PB_NAME, "wb")
f.write(net.SerializeToString())
f.close()
print("Done.")

'''
print("Generating json")
for key in export_data.keys():
    export_data[key] = export_data[key].tolist()
with open(MODEL+'.json', 'w') as outfile:
    json.dump(export_data, outfile)
'''
