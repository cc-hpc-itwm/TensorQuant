# TensorQuant

A TensorFlow toolbox for Deep Neural Network Quantization

Original paper: https://arxiv.org/abs/1710.05758

## Getting Started

### Structure

**Examples/** - Contains examples on how to use TensorQuant.

**Kernels/** - Contains C-files of the kernels used in the quantizers.

**Quantize/** - Contains the quantizers and override mechanic of the TensorQuant toolbox.

### Prerequisites

- [TensorFlow](https://www.tensorflow.org/) 2.0 (Keras)
- [Python](https://www.python.org/) 3.6

### Installing

Add the TensorQuant directory to your PYTHONPATH environment variable, so it can be found by your project.
``` shell
export PYTHONPATH=${PYTHONPATH}:<path-to-TensorQuant>
```

Compile the Kernels in the "Kernels/" directory. A makefile is provided (run 'make all'). There might be issues with the -D_GLIBCXX_USE_CXX11_ABI=0 flag. See [this link](https://www.tensorflow.org/extend/adding_an_op) (under 'Build the op library') for more help on this topic.

## Quantizing a Neural Network

TensorQuant temporarily hijacks the Keras layer identifiers in order to inline additional ops for the Quantization.
The override needs to be applied before the model is build. Therefore, TensorQuant cannot be used if the model is loaded from a container file (i.e. no calls to the "tf.keras.layers" classes).
In order to apply the overrides, you must import the "override" module from TensorQuant in your main file:

``` python
from TensorQuant.Quantize import override
```

The layers for quantization are selected via a Dictionary, which maps layer names to quantizers. The available quantizers are in "TensorQuant.Quantize.Quantizers".

For example, you can provide a dictionary in your python code like this:
``` python
override.extr_q_map={"Conv1" : tensorQuant.Quantize.Quantizers.FixedPointQuantizer_nearest(16,8)}
```

Alternatively, you can provide a .json file
```json
{
    "Layer_name" : "Quantizer_shortcut_string"
}
```
The quantizer shortcut strings are defined in the same file in the "quantizer_selector" function (e.g. "nearest,16,8" would create a fixed point quantization with 32bits and 16bit fractional part).

The layer names do not require to match the real names entirely, but every layer which contains a matching substring will be quantized with the given quantizer. This allows to quantize entire blocks of layers. As of writing this readme, there is an issue with the "tf.name_scope" feature together with Keras layers, so it is not a reliable way to structure your network.

Load the json file with
```python
from TensorQuant.Quantize import utils

override.extr_q_map = utils.quantizer_map(json_filename)
```
The available Quantizer shortcut strings are in the file "TensorQuant.Quantize.utils.quantizer_selector".

Currently, there is "extr_q_map" for layer activations and "weight_q_map" for the layer weights. "intr_q_map" for intrinsic quantization is not available in this version of TensorQuant.
Set these dictionaries before the model is build (i.e. before calling tensorflow.keras.layers classes). If you do not want to use quantization, set the quantizer map to "None" (default).

There are no changes required in your Keras model. However, the override mechanic is very sensitive to the exact identifiers of the classes, so it might be necessary to use the full identifiers (e.g. "tf.keras.layers.Conv2D"), or to create aliases (e.g. "Conv2D = tf.keras.layers.Conv2D"), as shown in the LeNet example.
``` python
# Introducing an alias for a Keras layer
Convolution2D = tf.keras.layers.Convolution2D

model = tf.keras.models.Sequential()

model.add(Convolution2D(
    filters = 20,
    kernel_size = (5, 5),
    padding = "same",
    input_shape = (28, 28, 1),
    activation="relu",
    name="Conv1"))
```

## Overriding Layers
The number of initially available overrides does not span the complete set of available Keras layers. Any Keras layer can be hijacked, like in this example:
``` python
import tensorflow as tf
from TensorQuant.Quantize.override_functions import generic_keras_override

keras_conv2d = tf.keras.layers.Conv2D
keras_conv2d_override = generic_keras_override(keras_conv2d)
# the override happens in this line
tf.keras.layers.Conv2D = keras_conv2d_override
# optionally, override any aliases
tf.keras.layers.Convolution2D = keras_conv2d_override
```
The overrides must be done before the model is build. The available overrides are in "TensorQuant.Quantize.override". Additional overrides can be placed in that file as well.

## Authors

Dominik Loroch (Fraunhofer ITWM)

Please reference to [this](https://arxiv.org/abs/1710.05758) paper.
