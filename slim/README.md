# Slim
This file contains modified slim models, which can be used with TensorQuant.

## Structure
This file contains:

**datasets/** - Utilities for loading the input images.

**deployment/** - Utilities used by the train_image_classifier.py script.

**nets/** - Contains the network descriptions and the nets_factory.py script.
The layer factories are called within the nets_factory.py script.

**preprocessing/** - Image preprocessing.

**scripts/** - Contains scripts to run different nets. The script invokes 'eval_image_classifier.py' and/or 'train_image_classifier.py'.

**misc/** - Utilities used by some python scripts.

**eval_image_classifier.py** - Central script executed for inference. It builds the network, takes care of feeding inputs and evaluates the output.

**train_image_classifier.py** - Central scipt executed for learning. It builds the network, takes care of feeding inputs and saves the trained parameters.

## Test the Slim Framework

To check if the toolbox works correctly, run a script from the scripts/ directory:
```
./scripts/infer_inceptionv1.sh &
```
It is necessary to set the directories for the dataset and the checkpoint files in the scripts. Always start the script from the slim/ directory.

## Adding New Networks to Slim

You can add your own network files to the existing slim/ environment. First, put the python file describing the topology into the nets/ folder. The new network needs to be registered:

- Add an entry to the 'preprocessing_fn_map' dictionary in the 'preprocessing/preprocessing_factory.py' script.

- Similarly, add an entry in the 'networks_map' and 'arg_scopes_map' dictionaries in the 'nets/nets_factory.py' script.

- Your network description should be wrapped in a function which takes at least one argument (the input images) and returns the tuple (last output layer, list of endpoints). See any other network in 'nets/' as an example.

- Additionally, if preparing for quantization, you should add
```
def myNet(images, ...
          **kwargs):
          conv2d = kwargs['conv2d']
          max_pool2d = kwargs['max_pool2d']
          ...
```
to the model.

- The layer calls in your model description must be equal to the layer factories taken from the kwargs. For example:
```
net = conv2d(images, 32, [5, 5], scope='conv1')
net = max_pool2d(net, [2, 2], 2, scope='pool1')
net = conv2d(net, 64, [5, 5], scope='conv2')
```
Notice that the easiest way of preparing a slim-based script for quantization is to remove the 'slim.' in front of every layer call. (Be careful not to remove slim. in front of other calls, like slim.arg_scope!)

The 'eval/train_image_classifier.py' scripts are now able to process your network and to apply quantization to it.


## Applying Quantization

Quantization is applied by passing dictionaries which map scope names of the layers to the desired quantizers. One dictionary can be provided for intrinsic, extrinsic and weight quantization, for example:
```
myDict={
            'myNet/conv1' : quantizer1
            'max_pool2d' : quantizer2
}
```
Notice that it is not necessary to specify the full scope name to quantize a layer. If a key in the dictionary fits multiple scope names, all of them will be quantized with the specified quantizer. This is a convenient method to quantize many layers simultaneously.
Another method is provided by the utility function:
```
q_map=TensorQuant.Quantize.utils.quantizer_map('qMap_filename.json')
```
It produces a dictionary according to the specified .json file 'qMap_filename.json'. The file can look like this:
```
{
        'myNet/conv1' : 'nearest,16,8'
        'max_pool2d' : 'zero,8,4'
}
```
Available quantizers are listed in the utility function 'quantizer_selector' in 'TensorQuant.Quantize.utils'.

If you do not want to use quantization, set the dictionaries to be None.
