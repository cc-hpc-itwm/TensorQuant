import tensorflow as tf

##################################
### Keras Layer implementation ###
##################################
# tensorflow/python/keras/layers.py
def create_qLayer(layer, intr_quantizer=None, extr_quantizer=None, weight_quantizer=None, name=None):
    class QLayer(tf.keras.layers.Layer):

        def __init__(self, name, config):
            super(self.__class__, self).__init__(**config)
            self.intr_quantizer = intr_quantizer
            self.extr_quantizer = extr_quantizer
            self.weight_quantizer = weight_quantizer

            #if name is None:
            #    name = "Quantized_%s" % self.__class__.__base__.__name__
            #self._name = name
            

        def build(self, input_shape):
            super(self.__class__, self).build(input_shape)
            # quantize weights
            if self.weight_quantizer is not None:
                weight_list = [w.name for w in self.weights]
                weight_list = [w.split("/")[-1][:-2] for w in weight_list] #[:-2] to remove ':0'
                for w in weight_list:
                    with tf.name_scope(w+"/quant"):
                        setattr(self, w, self.weight_quantizer(getattr(self, w)))
                        #tf.add_to_collection("quantized_variables", getattr(self, w))


        def call(self, inputs):
            output = super(self.__class__, self).call(inputs)
            if self.extr_quantizer is not None:
                with tf.name_scope("output/quant"):
                    output = self.extr_quantizer(output)
                    #tf.add_to_collection("quantized_outputs", output)
            return output

    cls = type(layer.__class__.__name__, (layer.__class__,),
               dict(QLayer.__dict__))
    return cls(name, layer.get_config())
