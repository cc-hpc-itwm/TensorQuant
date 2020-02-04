#import sys
#sys.path.append('..')

# LeNet for MNIST using Keras and TensorFlow
import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
import numpy as np

import model

# Add this for TensorQuant
from TensorQuant.Quantize import override

def main():

    # TensorQuant
    # Make sure the overrides are set before the model is created!
    # QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ
    override.extr_q_map={"Conv1" : "nearest,12,11"}
    override.weight_q_map={ "Conv1" : "nearest,32,16", "Dense3" : "nearest,32,16"}
    # QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ

    # Download the MNIST dataset
    dataset = mnist.load_data()

    train_data = dataset[0][0]
    train_labels = dataset[0][1]

    test_data = dataset[1][0]
    test_labels = dataset[1][1]

    # Reshape the data to a (70000, 28, 28, 1) tensord
    train_data = train_data.reshape([*train_data.shape,1]) / 255.0

    test_data = test_data.reshape([*test_data.shape,1]) / 255.0

    # Tranform training labels to one-hot encoding
    train_labels = np.eye(10)[train_labels]

    # Tranform test labels to one-hot encoding
    test_labels = np.eye(10)[test_labels]

    lenet = model.LeNet()

    lenet.summary()

    optimizer = tf.keras.optimizers.SGD(lr=0.01)

    # Compile the network
    lenet.compile(
        loss = "categorical_crossentropy",
        optimizer = optimizer,
        metrics = ["accuracy"])

    # Callbacks
    callbacks_list=[]
    #callbacks_list.append(callbacks.WriteTrace("timeline_%02d.json"%(myRank), run_metadata) )

    # Train the model
    lenet.fit(
        train_data,
        train_labels,
        batch_size = 128,
        nb_epoch = 1,
        verbose = 1,
        callbacks=callbacks_list)

    # Evaluate the model
    (loss, accuracy) = lenet.evaluate(
        test_data,
        test_labels,
        batch_size = 128,
        verbose = 1)
    # Print the model's accuracy
    print("Test accuracy: %.2f"%(accuracy))
    
if __name__ == "__main__":
    main()
