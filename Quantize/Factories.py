import tensorflow as tf
import sys
sys.path.append('../TensorLib')
from Quantize import QConv
from Quantize import QFullyConnect
from Quantize import QAvgPool

slim = tf.contrib.slim


def conv2d_factory(intr_q_map=None, extr_q_map=None):
    def func(*args,**kwargs):
        layer_ID=tf.get_variable_scope().name + "/" + kwargs["scope"]
        
        # If there is no intrinsic quantizer given, no layer is quantized.
        # If "conv2d" appears in the quantization list, all convolution layers are quantized.
        # if a string in intr_layers appears in the current variable scope or the scope, the layer is quantized.
        # Else, the layer is not quantized.
        if intr_q_map is None:
            net=slim.conv2d(*args,**kwargs)
        elif ("conv2d" in intr_q_map.keys() or
                   any(layer in layer_ID for layer in intr_q_map.keys()) ):
            layer = next(layer for layer in intr_q_map.keys() if(
                        layer in layer_ID or layer=="conv2d") )
            print( 'Quantizing (intr) layer %s'%(layer_ID) )
            kwargs['quantizer']=intr_q_map[layer]
            net=QConv.conv2d(*args,**kwargs) #, quantizer=intr_q_map[layer])
        else:
            net=slim.conv2d(*args,**kwargs)

        # Extr quantizer is similar to intrinsic.
        if extr_q_map is None:
            return net
        elif ("conv2d" in extr_q_map.keys() or
                   any(layer in layer_ID for layer in extr_q_map.keys()) ):
            layer = next(layer for layer in extr_q_map.keys() if(
                        layer in layer_ID or layer=="conv2d") )
            print( 'Quantizing (extr) layer %s'%(layer_ID) )
            return extr_q_map[layer].quantize(net)
        else:
            return net   
    return func

def fully_connected_factory(intr_q_map=None, extr_q_map=None):
    def func(*args,**kwargs):
        layer_ID=tf.get_variable_scope().name + "/" + kwargs["scope"]
        
        # If there is no intrinsic quantizer given, no layer is quantized.
        # If "fully_connected" appears in the quantization list, all convolution layers are quantized.
        # if a string in intr_layers appears in the current variable scope or the scope, the layer is quantized.
        # Else, the layer is not quantized.
        if intr_q_map is None:
            net=slim.fully_connected(*args,**kwargs)
        elif ("fully_connected" in intr_q_map.keys() or
                   any(layer in layer_ID for layer in intr_q_map.keys()) ):
            layer = next(layer for layer in intr_q_map.keys() if(
                        layer in layer_ID or layer=="fully_connected") )
            print( 'Quantizing (intr) layer %s'%(layer_ID) )
            net=QFullyConnect.fully_connected(*args,**kwargs, quantizer=intr_q_map[layer])
        else:
            net=slim.fully_connected(*args,**kwargs)

        # Extr quantizer is similar to intrinsic.
        if extr_q_map is None:
            return net
        elif ("fully_connected" in extr_q_map.keys() or
                   any(layer in layer_ID for layer in extr_q_map.keys()) ):
            layer = next(layer for layer in extr_q_map.keys() if(
                        layer in layer_ID or layer=="fully_connected") )
            print( 'Quantizing (extr) layer %s'%(layer_ID) )
            return extr_q_map[layer].quantize(net)
        else:
            return net   
    return func

def max_pool2d_factory(intr_q_map=None, extr_q_map=None):
    def func(*args,**kwargs):
        layer_ID=tf.get_variable_scope().name + "/" + kwargs["scope"]
        
        # If there is no intrinsic quantizer given, no layer is quantized.
        # If "max_pool2d" appears in the quantization list, all convolution layers are quantized.
        # if a string in intr_layers appears in the current variable scope or the scope, the layer is quantized.
        # Else, the layer is not quantized.
        if intr_q_map is None:
            net=slim.max_pool2d(*args,**kwargs)
        elif ("max_pool2d" in intr_q_map.keys() or
                   any(layer in layer_ID for layer in intr_q_map.keys()) ):
            layer = next(layer for layer in intr_q_map.keys() if(
                        layer in layer_ID or layer=="max_pool2d") )
            print( 'Quantizing (intr) layer %s'%(layer_ID) )
            # there is no effective intrinsic quantization for max pooling. 
            # Its the same as in the extrinsic case
            net=intr_q_map[layer].quantize(slim.max_pool2d(*args,**kwargs))
        else:
            net=slim.max_pool2d(*args,**kwargs)

        # Extr quantizer is similar to intrinsic.
        if extr_q_map is None:
            return net
        elif ("max_pool2d" in extr_q_map.keys() or
                   any(layer in layer_ID for layer in extr_q_map.keys()) ):
            layer = next(layer for layer in extr_q_map.keys() if(
                        layer in layer_ID or layer=="max_pool2d") )
            print( 'Quantizing (extr) layer %s'%(layer_ID) )
            return extr_q_map[layer].quantize(net)
        else:
            return net   
    return func

def avg_pool2d_factory(intr_q_map=None, extr_q_map=None):
    def func(*args,**kwargs):
        layer_ID=tf.get_variable_scope().name + "/" + kwargs["scope"]
        
        # If there is no intrinsic quantizer given, no layer is quantized.
        # If "avg_pool2d" appears in the quantization list, all convolution layers are quantized.
        # if a string in intr_layers appears in the current variable scope or the scope, the layer is quantized.
        # Else, the layer is not quantized.
        if intr_q_map is None:
            net=slim.avg_pool2d(*args,**kwargs)
        elif ("avg_pool2d" in intr_q_map.keys() or
                   any(layer in layer_ID for layer in intr_q_map.keys()) ):
            layer = next(layer for layer in intr_q_map.keys() if(
                        layer in layer_ID or layer=="avg_pool2d") )
            print( 'Quantizing (intr) layer %s'%(layer_ID) )
            net=QAvgPool.avg_pool2d(*args,**kwargs, quantizer=intr_q_map[layer])
        else:
            net=slim.avg_pool2d(*args,**kwargs)

        # Extr quantizer is similar to intrinsic.
        if extr_q_map is None:
            return net
        elif ("avg_pool2d" in extr_q_map.keys() or
                   any(layer in layer_ID for layer in extr_q_map.keys()) ):
            layer = next(layer for layer in extr_q_map.keys() if(
                        layer in layer_ID or layer=="avg_pool2d") )
            print( 'Quantizing (extr) layer %s'%(layer_ID) )
            return extr_q_map[layer].quantize(net)
        else:
            return net   
    return func

