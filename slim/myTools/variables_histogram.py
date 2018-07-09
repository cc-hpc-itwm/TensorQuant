"""
Make a histogram of all variables.

Author: Dominik Loroch
"""

import tensorflow as tf
from TensorQuant.Quantize import Quantizers
import json
import matplotlib.pyplot as plt
from misc import utils
import numpy as np

slim = tf.contrib.slim


MODEL='alexnet-model/logarithmic_long_training'

if tf.gfile.IsDirectory(MODEL):
      checkpoint_path = tf.train.latest_checkpoint(MODEL)
else:
      checkpoint_path = MODEL

plot_data=[]
export_data={}
with tf.Session() as sess:
    print('Opening %s'%(checkpoint_path))
    saver = tf.train.import_meta_graph(MODEL+'/model.meta')
    saver.restore(sess, checkpoint_path)
    print('Restorable variables:')
    variables = utils.get_all_variables_as_single_op('weights').eval()
    for var in slim.get_variables_to_restore():
        print('> %s'%(var.name))
        export_data[var.name]=var.eval().tolist()

#plt.hist(plot_data)            
#plt.show()
#print(export_data)

variables = abs(variables)
l_weights, n_weights = np.unique(variables,return_counts=True)
weight_hist=dict(zip(l_weights, n_weights))
#print(weight_hist)
plt.bar(list(weight_hist.keys()), weight_hist.values(),list(weight_hist.keys()))
plt.xscale('log', basex=2)
plt.show()

#with open(MODEL+'.json', 'w') as outfile:
#    json.dump(export_data, outfile)
