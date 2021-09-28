import tensorflow as tf
from Model_Creation import model
from Dataset_Creation import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model = model()

def train():
    epochs=30
    (x_train, y_train),(x_test,y_test),(x_valid,y_valid)=create()
    validation=(x_valid,y_valid)
    h=model.fit(x_train,y_train,epochs=epochs,validation_data=validation)
    pd.DataFrame(h.history).plot(figsize=(10,7))
    plt.grid(True)
    plt.show()
    model.save('model.h5')

def predict(img,sol):
    yp=np.argmax(model.predict(img),axis=-1)
    plt.imshow(img,cmap='binary')
    plt.axis('off')
    plt.show()



if __name__ == '__main__':
    train()
