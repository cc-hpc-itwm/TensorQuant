# Quantize
This file contains the TensorQuant toolbox.

## Structure

**Factories.py** - This utility is used to produce layers based on whether intrinsic, extrinsic or no quantization should be applied.

**FixedPoint.py** - Python wrappers for the fixed point quantization kernels.

**QAvgPool.py** - Reimplementation of the slim avg_pool2d layer (can be quantized intrinsically).

**QBatchNorm.py** - Reimplementation of the slim batch_norm layer (can be quantized intrinsically).

**QConv.py** - Reimplementation of the slim conv2d layer (can be quantized intrinsically).

**QFullyConnect.py** - Reimplementation of the slim fully_connected layer (can be quantized intrinsically).

**QSGD.py** - Reimplementation of the GradientDescentOptimizer (work in progress).

**Quantizers.py** - Definition of the Quantizer objects. They use the quantization kernels.

## Test

You can run any 'xx_test.py' script to verify if the reimplemented layers work properly, or start 'test_all.sh' to run all tests in a row. The message '---failed!---' will be displayed, if a test fails.
The tests generate a random input tensor and applies the reimplemented and the original slim layer to it. The outputs are compared element wise.

## Remarks

### Factories.py
- For each new layer type, a factory (a wrapper for 'generic_factory') must be declared. The wrapper passes the reimplemented and original slim layer to the generic function.
- Notice that the wrapper can perform additional computations, if necessary (see max_pool2d_factory for example).
- The 'generic_factory' returns a function which always has the interface of 'layer_function' (i.e. the quantizer is already passed as an argument).

### FixedPoint.py
- There are also some kernel-less functions, which are used for verification.

### QAvgPool.py
- Mostly an exact copy of the original slim layer (tensorflow/python/layers/pooling.py).
- The quantizer is passed as an argument to the layer call.
- The 'avg_pool' function performs the actual computation of the layer ( therefore, this is where the intrinsic quantization happens).

### QBatchNorm.py
- Mostly an exact copy of the original slim layer (tensorflow/contrib/layers/python/layers/layers.py).
- The quantizer is passed as an argument to the layer call. This layer has trainable variables (mean and variance), which will be also quantized if 'use_quantized_weights' is True.
- The 'qbatch_normalization' function performs the actual computation of the layer (also the location of intrinsic quantization).

### QConv.py
- Mostly an exact copy of the original slim layer (tensorflow/contrib/layers/python/layers/layers.py).
- The quantizer is passed as an argument to the layer call. This layer has trainable variables (weights), which will be also quantized if 'use_quantized_weights' is True.
- The 'q2dconvolution_op' function performs the actual computation of the layer (also the location of intrinsic quantization). Notice that this layer is computed with tf.while_loops. If there are issues with GPU memory, you can adjust 'parallel_iterations' and 'swap_memory' of the tf.while_loops.

### QFullyConnect.py
- Mostly an exact copy of the original slim layer (tensorflow/contrib/layers/python/layers/layers.py).
- The quantizer is passed as an argument to the layer call. This layer has trainable variables (weights), which will be also quantized if 'use_quantized_weights' is True.
- The 'qmatmul' function performs the actual computation of the layer (also the location of intrinsic quantization). There is no tf.while_loop here, but the batch is divided into several sub-networks. Therefore, the batch size should not be chosen high if fully connected layers are used.

### QSGD.py (work in progress)
- Mostly an exact copy of the original slim layer (tensorflow/python/training/gradient_descent.py).
- This optimizer will calculate and apply the updates in a fully quantized manner.

### Quantizers.py
- Every Quantizer implements the quantizer interface defined by 'Quantizer_if'. It takes a tensor as an argument and returns a quantized tensor.
- In most cases, the 'quantize' method calls the corresponding python wrapper of the kernel.
- There is a 'NoQuantizer' quantizer, which simply returns the unquantized tensor. This quantizer is used for debugging.
