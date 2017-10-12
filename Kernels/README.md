# Kernels
This folder contains the C-kernels used by the quantizers.

## Structure
**compile.sh** - This script can be used to compile a single C-file.

**Makefile** - This makefile is set up to compile all files ending in '.cc' as TensorFlow kernels.

**KernelName.cc** - The C-kernels.

## Compilation
Simply run
```
make all
```
in this folder to compile all kernels. There can be issues with the compiler flags ( especially -D_GLIBCXX_USE_CXX11_ABI=0; see [this link](https://www.tensorflow.org/extend/adding_an_op) (under 'Build the op library') for more help on this topic.).

Single kernels can also be compiled with:
```
source compile.sh kernel_name
```
Pass the kernel file without the filename extension (.cc)!

## Details
A description on how to implement kernels is given [here](https://www.tensorflow.org/extend/adding_an_op).
