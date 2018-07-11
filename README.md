# TensorQuant

A TensorFlow toolbox for Deep Neural Network Quantization

Original paper: https://arxiv.org/abs/1710.05758

## Getting Started

### Structure

**slim/** - Contains a modified version of the [Slim](https://github.com/tensorflow/models/tree/master/research/slim) model library.

**Kernels/** - Contains C-files of the kernels used in the quantizers.

**Quantize/** - Contains the quantizers and layer factories of the TensorQuant toolbox.

**Optimize/** - Contains custom optimizers.

### Prerequisites

- [TensorFlow](https://www.tensorflow.org/) 1.7
- [Python](https://www.python.org/) 2.7 / 3.6

### Installing

Add the TensorQuant directory to your PYTHONPATH environment variable.
```
export PYTHONPATH=${PYTHONPATH}:<path-to-TensorQuant>
```

Compile the Kernels within the Kernels/ directory. A makefile is provided (run 'make all'). There might be issues with the -D_GLIBCXX_USE_CXX11_ABI=0 flag. See [this link](https://www.tensorflow.org/extend/adding_an_op) (under 'Build the op library') for more help on this topic.

If you are planning to use slim/, make sure the datasets (e.g. MNIST and ImageNet) are already installed and set up (see [link](https://github.com/tensorflow/models/tree/master/research/slim) for help). The original slim model library comes with a set of pre-trained models, which can be used with TensorQuant. Make sure to download the checkpoint files from [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models). Put a file called 'checkpoint' into the same folder as the model .ckpt file with the content
```
model_checkpoint_path: "model-name.ckpt"
```
or TensorFlow will not be able to restore the model parameters from the specified file.

## Running Networks with the Slim Framework

You can use the scripts in the slim/scripts/ directory as a starting point. Make sure the directories in the scripts are set up correctly. 'TRAIN_DIR' should point to the directory with the .ckpt file and 'DATASET_DIR' to the dataset. Run a script (e.g. GoogLeNet on ImageNet) from the slim/ directory with
```
./scripts/infer_inceptionv1.sh &
```
to see if the toolbox is set up properly.

See the READMEs in the subfolders to see further information on the toolbox.

## Using Quantization Outside of the Slim Framework

The quickest way to use quantization in any topology outside of the provided slim/ example is to define a file (e.g. called tq_layers.py) like this:

```
# tq_layers.py
import tensorflow as tf
import TensorQuant as tq
import os

from TensorQuant.Quantize import *
from TensorQuant.Quantize.Factories import generic_factory
from TensorQuant.Quantize.Factories import extr_only_generic_factory
from TensorQuant.slim.utils import *
from local_settings import *


# quantization maps passed to factories, layer call functions for use in model files
#-----------------------------------------------------------------
intr_file='intr_map.json'
extr_file='extr_map.json'
weights_file = 'weights_map.json'


if os.path.exists(intr_file):
    intr_q_map = quantizer_map(intr_file)
else:
    intr_q_map = None

if os.path.exists(extr_file):
    extr_q_map = quantizer_map(extr_file)
else:
    extr_q_map = None

if os.path.exists(weights_file):
    weight_q_map = quantizer_map(weights_file)
else:
    weight_q_map = None


#conv2d = tf.contrib.layers.conv2d
conv2d = generic_factory(tf.contrib.layers.conv2d, QConv.conv2d, 
                           intr_q_map=intr_q_map, extr_q_map=extr_q_map, 
                           weight_q_map=weight_q_map)

add = extr_only_generic_factory(tf.add, 
                           intr_q_map=intr_q_map, extr_q_map=extr_q_map)

max_pool2d = extr_only_generic_factory(tf.contrib.layers.max_pool2d, 
                           intr_q_map=intr_q_map, extr_q_map=extr_q_map)


resize_bilinear = extr_only_generic_factory(tf.image.resize_bilinear, 
                           intr_q_map=intr_q_map, extr_q_map=extr_q_map)
```

You can then add

```
from tq_layers import *
```

to your model file. Change the layer function names in the model file like in the slim framework example and your model will be quantized with the quantizer map .json-files as defined in tq_layers.py. It is not necessary to add kwargs to your model header like in the slim framework, since the layer factories are loaded with the import. Notice that your model will always be quantized with the same quantizer map files, if they are present.


## Authors

Dominik Loroch (Fraunhofer ITWM)

Please reference to [this](https://arxiv.org/abs/1710.05758) paper.
