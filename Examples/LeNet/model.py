import tensorflow as tf

def LeNet():
    # TensorQuant is sensitive to the exact identifiers.
    # It is advised to use the full name ('tf.keras.layers.SomeLayer') or use aliases like shown here.
    Convolution2D = tf.keras.layers.Convolution2D
    MaxPooling2D = tf.keras.layers.MaxPooling2D
    Flatten = tf.keras.layers.Flatten
    Dense = tf.keras.layers.Dense

    model = tf.keras.models.Sequential()

    with tf.name_scope("LeNet"):
        with tf.name_scope("Convolution_Block"):
            # Add the first convolution layer
            model.add(Convolution2D(
                filters = 20,
                kernel_size = (5, 5),
                padding = "same",
                input_shape = (28, 28, 1),
                activation="relu",
                name="Conv1"))

            # Add a pooling layer
            model.add(MaxPooling2D(
                pool_size = (2, 2),
                strides =  (2, 2),
                name="MaxPool1"))

            # Add the second convolution layer
            model.add(Convolution2D(
                filters = 50,
                kernel_size = (5, 5),
                padding = "same",
                activation="relu",
                name="Conv2"))

            # Add a second pooling layer
            model.add(MaxPooling2D(
                pool_size = (2, 2),
                strides = (2, 2),
                name="MaxPool2"))

        # Flatten the network
        model.add(Flatten())

        with tf.name_scope("Dense_Block"):
            # Add a fully-connected hidden layer
            model.add(Dense(500,
                activation="relu",
                name="Dense3"))

            # Add a fully-connected output layer
            model.add(Dense(10,
                activation="softmax",
                name="Dense4"))
    return model
