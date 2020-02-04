import tensorflow as tf
from TensorQuant.Quantize.override_functions import generic_keras_override

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# override these qmaps in your main application
# Example:
# intr_q_map = {    "MyNetwork/Conv2D_1" : "nearest,32,16",
#                   "MyNetwork/Conv2D_2" : "nearest,16,8"}

intr_q_map=None
extr_q_map=None
weight_q_map=None
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 'tensorflow.keras.layers.Convolution2D' override
keras_conv2d = tf.keras.layers.Conv2D
keras_conv2d_override = generic_keras_override(keras_conv2d)
tf.keras.layers.Conv2D = keras_conv2d_override
tf.keras.layers.Convolution2D = keras_conv2d_override

# 'tf.keras.layers.Conv1D' override
keras_conv1d = tf.keras.layers.Conv1D
keras_conv1d_override = generic_keras_override(keras_conv1d)
tf.keras.layers.Conv1D = keras_conv1d_override

# 'tf.keras.layers.Dense' override
keras_dense = tf.keras.layers.Dense
keras_dense_override = generic_keras_override(keras_dense)
tf.keras.layers.Dense = keras_dense_override

# 'tf.keras.layers.MaxPooling2D' override
keras_maxpool2d = tf.keras.layers.MaxPooling2D
keras_maxpool2d_override = generic_keras_override(keras_maxpool2d)
tf.keras.layers.MaxPooling2D = keras_maxpool2d_override

# 'tf.keras.layers.MaxPool1D' override
keras_maxpool1d = tf.keras.layers.MaxPool1D
keras_maxpool1d_override = generic_keras_override(keras_maxpool1d)
tf.keras.layers.MaxPool1D = keras_maxpool1d_override
