import tensorflow as tf
import time
import os

def create_model(lf,opt,met,num):
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
    model.save(path)