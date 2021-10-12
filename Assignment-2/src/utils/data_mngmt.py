import tensorflow as tf


def get_data(size):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test,y_test)=mnist.load_data()
    x_valid,x_train=x_train[:size] / 255., x_train[size:] / 255.
    y_valid, y_train = y_train[:size], y_train[size:]
    x_test=x_test/255.
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
