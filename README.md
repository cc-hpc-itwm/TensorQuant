# TensorQuant

A TensorFlow toolbox for Deep Neural Network Quantization

Original paper: https://arxiv.org/abs/1710.05758

## Getting Started

### Structure

**slim/** - Contains a modified version of the [Slim](https://github.com/tensorflow/models/tree/master/research/slim) model library.

**Kernels/** - Contains C-files of the kernels used in the quantizers.

**Quantize/** - Contains the quantizers and layer factories of the TensorQuant toolbox.

### Prerequisites

- [TensorFlow](https://www.tensorflow.org/) 1.0 or higher.
- [Python](https://www.python.org/) 2.7 / 3.6 or higher

### Installing

Add the TensorQuant directory to your PYTHONPATH environment variable.
```
export PYTHONPATH=${PYTHONPATH}:<path-to-TensorQuant>/TensorQuant
```

Compile the Kernels within the Kernels/ directory. A makefile is provided (simply run 'make all'). There might be issues with the -D_GLIBCXX_USE_CXX11_ABI=0 flag. See [this link](https://www.tensorflow.org/extend/adding_an_op) (under 'Build the op library') for more help on this topic.

If you are planning to use slim/, make sure the datasets (e.g. MNIST and ImageNet) are already installed and set up (see [link](https://github.com/tensorflow/models/tree/master/research/slim) for help). The original slim model library comes with a set of pre-trained models, which can be used with TensorQuant. Make sure to download the checkpoint files from [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models). Put a file called 'checkpoint' into the same folder as the model .ckpt file with the content
```
model_checkpoint_path: "model-name.ckpt"
```
or TensorFlow will not be able to restore the model parameters from the specified file.

## Running Simulations with Slim

You can use the scripts in the slim/scripts/ directory as a starting point. Make sure the directories in the scripts are set up correctly. 'TRAIN_DIR' should point to the directory with the .ckpt file and 'DATASET_DIR' to the dataset. Run a script (e.g. GoogLeNet on ImageNet) from the slim/ directory with
```
./scripts/infer_googlenet_on_imagenet.sh &
```
to see if the toolbox is set up properly (this test should take only a minute, because only a few inputs are calculated). If it works properly, you should see an accuracy (the value does not matter, because only a few images are tested).

See the READMEs in the subfolders to see further information on the toolbox.

## Authors

Dominik Loroch (Fraunhofer ITWM)

Please reference to [this](https://arxiv.org/abs/1710.05758) paper.
