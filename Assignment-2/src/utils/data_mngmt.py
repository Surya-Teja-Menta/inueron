import tensorflow as tf
import logging


log_file='logs/general_logs/training.log'
log_format='%(levelname)s: %(asctime)s: %(message)s'
logging.basicConfig(filename=log_file,level=logging.INFO,format=log_format)


def get_data(size):
    logging.info('>>>>>Started extracting data:')
    mnist = tf.keras.datasets.mnist
    logging.info('>>>>>Started Splitting data:')

    (x_train, y_train),(x_test,y_test)=mnist.load_data()
    x_valid,x_train=x_train[:size] / 255., x_train[size:] / 255.
    y_valid, y_train = y_train[:size], y_train[size:]
    x_test=x_test/255.
    logging.info('>>>>>Shapes')
    logging.info((x_train.shape, y_train.shape), (x_valid.shape, y_valid.shape), (x_test.shape, y_test.shape))
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
