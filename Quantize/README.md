# Quantize
This file contains the TensorQuant toolbox.

## Structure

**Quantizers.py** - Definition of the Quantizer objects. Use these in your quantizer maps.

**utils.py** - Utilities to generate quantization maps.

**Factories.py** - This utility is used to produce layers based on whether intrinsic, extrinsic or weight quantization should be applied.

**FixedPoint.py** - Python wrappers for the fixed point quantization kernels.

**QuantKernelWrapper.py** - Python wrappers for non-fixed point quantization kernels.

**QLayer.py** - Reimplementation of a slim layer (can be quantized intrinsically).

**test_all.sh** - Run a test for quantizer and layer implementations.


## Test

You can run any 'xx_test.py' script to verify if the reimplemented layers work properly, or start 'test_all.sh' to run all tests in a row. The message '+++passed!+++' will be displayed if successful. The the scripts for details on the tests.

## Remarks

### Factories.py
- 'generic_facotry' allows for intrinsic quantization. For layers which do not need intrinsic quantization and which have no weights, the 'extr_only_generic_factory' can be used.
- For each new layer type, a factory (a wrapper for 'generic_factory' or 'extr_only_generic_factory') must be declared. The wrapper passes the reimplemented and original slim layer to the generic function.
- Notice that the wrapper can perform additional operations, if necessary.
- The factories return a function which always has the same interface as 'layer_function' (i.e. the quantizer is already passed as an argument).

### FixedPoint.py
- There are also some functions without C-Kernels, which are used for verification.

### Quantizers.py
- Every Quantizer implements the quantizer interface defined by 'Quantizer_if'. It takes a tensor as an argument and returns a quantized tensor.
- In most cases, the 'quantize' method calls the corresponding python wrapper of the kernel.
- There is a 'NoQuantizer' quantizer, which simply returns the unquantized tensor. This quantizer is used for debugging.
