import tensorflow as tf
from TensorQuant.Quantize.QLayer import create_qLayer
from TensorQuant.Quantize import utils as tq_utils
from TensorQuant.Quantize import override

# This function creates an override function for nearly all keras layers with weights.
def generic_keras_override(Class):
    print("Override for Class %s is active."%(Class.__module__+ "." + Class.__name__))
    def override_func(*args, **kwargs):
        def find_quantizer(name, _map):
            _map = tq_utils.quantizer_map(_map)
            if _map is None:
                return None
            quantizer = None
            for name in _map.keys():
                if(name in layer_ID):
                    quantizer = _map[name]
                    break
            return quantizer

        # if "name" in kwargs.keys():
        #     name = "/"+kwargs["name"]
        # else:
        #     name = ''

        layer = Class(*args, **kwargs)

        # Tensorflow 1.X
        try:
            layer_scope = tf.get_default_graph().get_name_scope()
        except:
            layer_scope = ""

        if layer_scope !="":
            layer_ID = tf.get_default_graph().get_name_scope() + "/" + layer.name
        else:
            layer_ID = layer.name 

        intr_quantizer = find_quantizer(layer_ID, override.intr_q_map)
        extr_quantizer = find_quantizer(layer_ID, override.extr_q_map)
        weight_quantizer = find_quantizer(layer_ID, override.weight_q_map)
        
        layer=create_qLayer(layer,
                            intr_quantizer=intr_quantizer,
                            extr_quantizer=extr_quantizer,
                            weight_quantizer=weight_quantizer)
        # Info about quantized Layers
        if intr_quantizer is not None:
            print("%s: internally quantized with %s"%(layer.name, intr_quantizer))
        if extr_quantizer is not None:
            print("%s output quantized with %s"%(layer.name, extr_quantizer))
        if weight_quantizer is not None:
            print("%s weights quantized with %s"%(layer.name, weight_quantizer))
        return layer
    return override_func
