# Quantize
This directory contains the TensorQuant core mechanics.

## Structure

**Quantizers.py** - Definition of the Quantizer objects. Use these in your quantizer maps.

**utils.py** - Utilities to generate quantization maps.

**override.py** - defines the active overrides and the global quantization dictionaries.

**FixedPoint.py** - Python wrappers for the fixed point quantization kernels.

**QuantKernelWrapper.py** - Python wrappers for non-fixed point quantization kernels.

**QLayer.py** - Defines a generic Keras Layer with enabled quantization.

**override_functions.py** Contains the "generic_keras_override" function, which decides which layers are to be hijacked.

## Remarks

### Quantizers.py
- Every Quantizer implements the quantizer interface defined by 'Quantizer_if'. It takes a tensor as an argument and returns a quantized tensor. Use this interface to add additional quantizers.
- There is a 'NoQuantizer' quantizer, which simply returns the unquantized tensor. This quantizer is used for debugging.
- Most of the quantizers are implemented with Tensorflow layers. The C-Code kernels can be called (but often not used) with the "C_quantize" method. In fact, if the compilation of the C-code kernels should not work, one can work around that step.
- It is possible to define custom gradients for the Quantizers ("def grad(dy):"). The gradients in the TensorQuant quantizers are initially straight through, but can be modified.

### utils.py
- the function "quantizer_map" can be used to generate quantizer maps. The function takes either a .json file, or a dictionary with following structure:
```json
{
    "Layer_name" : "Quantizer_shortcut_string"
}
```
The quantizer shortcut strings are defined in the same file in the "quantizer_selector" function (e.g. "nearest,32,16" would create a fixed point quantization with 32bits and 16bit fractional part).
The layer names do not require to match the real names entirely, but every layer which contains a matching substring will be quantized with the given quantizer. This allows to quantize entire blocks of layers. As of writing this readme, there is an issue with the "tf.name_scope" feature together with Keras layers, so it is not a reliable way to structure your network.

### override.py
- The available overrides do not cover all Keras layers. However, the overrides can be easily extended for other Keras layers with:
``` python
keras_SomeLayer = tf.keras.layers.SomeLayer
keras_SomeLayer_override = generic_keras_override(keras_SomeLayer)
# override the Keras layer
tf.keras.layers.SomeLayer = keras_SomeLayer_override
# optional: override for aliases
tf.keras.layers.SomeLayer_alias = keras_conv2d_override
```
- the "intr_q_map" has no effect in this version of TensorQuant
- the "extr_q_map" (for activations) and "weight_q_map" (for weights and biases) dictionaries can be written with a dictionary defining the desired quantization setup.

