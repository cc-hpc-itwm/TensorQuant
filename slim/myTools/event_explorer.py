import tensorflow as tf
import sys, os
sys.path.append('/home/loroch/TensorFlow/TensorLib')
from Quantize import *
import matplotlib as mplot
import matplotlib.pyplot as plt
import datetime
import json

#NAME='alexnet_32-16'
PATH="./alexnet-model/width_16_14/"
FILES=[]
filelist=os.listdir(PATH)
filelist=sorted(filelist)
for f in filelist:
    if '.tfevents' in f:
        FILES.append(f)

#FILE='events.out.tfevents'
#var_name='total_loss'
var_name='gradient'
var_name2='weights'

data={}
data['step']=[]
data['avg_min']=[]
data['avg_max']=[]
minimum=0
maximum=0
for f in FILES:
    print('Opening file %s'%f)
    try:
        for e in tf.train.summary_iterator(PATH+f):
          avg_min=0
          avg_max=0
          count=0
          for v in e.summary.value:
              #print('%s: %.4f'%(v.tag,v.simple_value))
              if var_name in v.tag and var_name2 in v.tag:
                  count+=1
                  avg_min+=v.histo.min
                  avg_max+=v.histo.max
                  minimum=min(minimum,v.histo.min)
                  maximum=max(maximum,v.histo.max)
                  #print('%s: %.12f  to %.12f'%(v.tag,v.histo.min, v.histo.max))
                  #print(v.histo)
          #print('Step %d: Minimum: %.4f  Maximum: %.4f'%(e.step, minimum, maximum))
          if count!=0:
            data['step'].append(e.step)
            data['avg_min'].append(avg_min/count)
            data['avg_max'].append(avg_max/count)
          
    except Exception as e:
          print(str(e))
    print('Minimum: %.4f  Maximum: %.4f'%(minimum, maximum))

      
# Plot data
fig = plt.figure(1)
plt.clf()
mplot.rcParams.update({'font.size': 14})
#patches = []

#plt.semilogy(data['step'],data['avg_min'],label='min')
#plt.semilogy(data['step'],data['avg_max'],label='max')
plt.plot(data['step'],data['avg_min'],label='min')
plt.plot(data['step'],data['avg_max'],label='max')

# axes 
axes = plt.gca()
#axes.set_xlim([-1,1])
axes.tick_params(which='both', direction='in')
plt.grid(b=True, which='major', color='0.9', linestyle='--')

# text
plt.xlabel('step')
plt.ylabel('a.u.')
#plt.title()
plt.legend(loc='lower right')
#plt.legend(handles=patches,loc='center right')

#plt.show()
#fig_savename=str(datetime.datetime.now())+'_Output.jpg'
fig_savename='Output.jpg'
print('Saving figure %s'%fig_savename)
fig.savefig(fig_savename, dpi=300, format='jpg')


