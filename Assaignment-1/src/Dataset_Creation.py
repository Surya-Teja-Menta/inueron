import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

LOG_FILE = 'logs/Training.log'
logging_format = '%(levelname)s: %(asctime)s: %(message)s'

logging.basicConfig(filename=LOG_FILE,level=logging.INFO,format=logging_format)



def create():
    m=tf.keras.datasets.mnist
    (x_train_f, y_train_f),(x_test, y_test)=m.load_data()
    x_valid,x_train=x_train_f[50000:]/255,x_train_f[:50000]/255
    y_valid,y_train=y_train_f[50000:],y_train_f[:50000]
    x_test=x_test/255
    logging.info([(x_train.shape, y_train.shape),(x_test.shape, y_test.shape),(x_valid.shape, y_valid.shape)])
    return (x_train, y_train),(x_test, y_test),(x_valid, y_valid)

def ims(img):
    i=img
    plt.imshow(img,cmap='binary')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    create()
