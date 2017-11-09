# slim
This file contains modified slim models, which can be used with TensorQuant.

## Structure
This file contains:

**datasets/** - Utilities for the input images (unmodified slim).

**deployment/** - Utilities used by the train_image_classifier.py script (unmodified slim).

**nets/** - Contains the network descriptions and the nets_factory.py script (modified slim).
The layer factories are called within the nets_factory.py script.

**preprocessing/** - Network dependent image preprocessing (unmodified slim)

**scripts/** - Contains scripts to run different nets (modified slim). The script invokes 'eval_image_classifier.py' and/or 'train_image_classifier.py'.

**eval_image_classifier.py** - Central script executed for inference (modified slim). It builds the network, takes care of feeding inputs and evaluates the output.

**train_image_classifier.py** - Central scipt executed for learning (modified slim). It builds the network, takes care of feeding inputs and saves the trained parameters.

**utils.py** - Additional utilities used for applying quantization. Used in the 'xxx_image_classifier.py' scripts in the same directory.

## Test

To check if the toolbox works correctly, run a script from the scripts/ directory:
```
./scripts/infer_googlenet_on_imagenet.sh &
```
The toolbox needs to be set up correctly before trying to run the script. Check the directories for the dataset and the models in the scripts! Always start the script from the slim/ directory.

## Adding new networks

You can add your own network files to the existing slim/ environment. First, the new network needs to be registered:

- Add an entry to the 'preprocessing_fn_map' dictionary in the 'preprocessing/preprocessing_factory.py' script.

- Similarly, add an entry in the 'networks_map' and 'arg_scopes_map' dictionaries in the 'nets/nets_factory.py' script.

- Your network description should be wrapped in a function which takes at least one argument (the input images) and returns the tuple (last output layer, list of endpoints). See any other network in 'nets/' as an example.

- Additionally, if preparing for quantization, you should pass the layers used by your network by introducing
```
def myNet(images, ...
          conv2d=slim.conv2d,
          max_pool2d=slim.max_pool2d,
          fully_connected = slim.fully_connected):
```
to the function header.

- The layer calls in your network description must be equal to the layers from the function header. For example:
```
net = conv2d(images, 32, [5, 5], scope='conv1')
net = max_pool2d(net, [2, 2], 2, scope='pool1')
net = conv2d(net, 64, [5, 5], scope='conv2')
```
Notice that the easiest way of preparing a slim-based script for quantization is to remove the 'slim.' in front of every layer call. (Be careful not to remove slim. in front of other calls, like slim.arg_scope!)

- In the 'network_fn' function in 'nets/nets_factory.py', add an entry with your network in the form:
```
if 'myNet' in name:
            return func(images, ...
                    conv2d=conv2d,
                    max_pool2d=max_pool2d,
                    avg_pool2d=avg_pool2d)
```

The 'eval/train_image_classifier.py' scripts are now able to process your network and to apply intrinsic and extrinsic quantization to it.

## Applying Quantization

Quantization is applied by passing dictionaries which map scope names of the layers to the desired quantizers, for each intrinsic and extrinsic quantization, for example:
```
myDict={
            'myNet/conv1' : quantizer1
            'max_pool2d' : quantizer2
}
```
Notice that it is not necessary to specify the full scope name to quantize a layer. If a key in the dictionary fits multiple scope names, all of them will be quantized with the specified quantizer. This is a convenient method to quantize many layers simultaneously.
Another method is provided by the utility function:
```
intr_q_map=utils.quantizer_map('qMap_filename')
```
It produces a dictionary according to the specified .json file 'qMap_filename'. The file can look like this:
```
{
        'myNet/conv1' : 'nearest,16,8'
        'max_pool2d' : 'zero,8,4'
}
```
Available quantizers are listed in the utility function 'quantizer_selector'.

If you do not want to use quantization, set the dictionaries to be None.

The dictionaries are passed to a function from 'nets/nets_factory.py':
```
network_fn = nets_factory.get_network_fn( ...,
        intr_q_map=intr_q_map,
        extr_q_map=extr_q_map)
```
Inside this function, they are passed to the layer factories, which take care of quantizing the layers.
