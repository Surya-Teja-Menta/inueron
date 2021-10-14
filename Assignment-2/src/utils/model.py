import tensorflow as tf
import time
import os
import logging


log_file='logs/general_logs/training.log'
log_format='%(levelname)s: %(asctime)s: %(message)s'
logging.basicConfig(filename=log_file,level=logging.INFO,format=log_format)

def create_model(lf,opt,met,num):
    logging.info('>>>>>Creating Layers')
    layers=[
        tf.keras.layers.Flatten(input_shape=[28,28],name='input_Layer'),
        tf.keras.layers.Dense(400,activation='relu'),
        tf.keras.layers.Dense(200,activation='relu'),
        tf.keras.layers.Dense(100,activation='relu'),
        tf.keras.layers.Dense(num,activation='softmax')

    ]
    model=tf.keras.models.Sequential(layers)
    model.summary()
    model.compile(loss=lf,optimizer=opt,metrics=met)
    return model

def get_filename(name):
    filename=time.strftime(f'%Y%m%d_%H%M%S_{name}')
    return filename
def save_model(model,name,dir):
    filename=get_filename(name)
    path=os.path.join(dir,filename)
    logging.info('>>>>>Saving the model at {} with name:{}'.format(path,filename))
    model.save(path)