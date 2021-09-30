import tensorflow as tf
import logging

LOG_FILENAME = 'logs/Training.log'
logging_format = '%(levelname)s: %(asctime)s: %(message)s'
logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO,format=logging_format)


def model():
    layers=[
        tf.keras.layers.Flatten(input_shape=[28,28]),
        tf.keras.layers.Dense(400,activation='relu'),
        tf.keras.layers.Dense(200,activation='relu'),
        tf.keras.layers.Dense(100,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
    ]
    model=tf.keras.models.Sequential(layers)
    logging.info(model.layers)
    logging.info(model.summary())
    Loss='sparse_categorical_crossentropy'
    optimizer='Adam'
    metrics=['accuracy']
    model.compile(loss=Loss,optimizer=optimizer,metrics=metrics)
    return model

if __name__ == '__main__':
    model()