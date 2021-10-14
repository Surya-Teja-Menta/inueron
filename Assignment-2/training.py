import os
from src.utils.common import read_config
from src.utils.data_mngmt import get_data
from src.utils.model import *
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import logging


log_file='logs/general_logs/training.log'
log_format='%(levelname)s: %(asctime)s: %(message)s'
logging.basicConfig(filename=log_file,level=logging.INFO,format=log_format)
def training(config):
    logging.info('>>>>>Configuring parameters')
    config=read_config(config)
    size=config['params']['size']
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_data(size)
    lf = config["params"]["loss_function"]
    opt = config["params"]["optimizer"]
    met = config["params"]["metrics"]
    num = config["params"]["num_classes"]
    tlog_dir='logs/tensorboard_logs/'

    model = create_model(lf, opt, met, num)
    epochs = config["params"]["epochs"]
    validation=(x_valid,y_valid)
    CKPT_path = "logs/model_ckpt.h5"
    tensorboard_cb=tf.keras.callbacks.TensorBoard(log_dir=tlog_dir)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)
    callbacks=[tensorboard_cb,early_stopping_cb,checkpointing_cb]
    logging.info('>>>>>Model Training')
    history=model.fit(x_train, y_train, epochs=epochs,validation_data=validation,callbacks=callbacks)
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    model_name = config["artifacts"]["model_name"]
    save_model(model, model_name, model_dir_path)
    results=model.evaluate(x_test,y_test,batch_size=64)
    logging.info('Test Loss, Test Accuracy are:{}'.format(results))
    his=pd.DataFrame(history.history).plot(figsize=(10,7))
    logging.info('\nHistory:\n {}'.format(pd.DataFrame(history.history)))
    
    filename=time.strftime(f'%Y%m%d_%H%M%S_plot.png')
    path=os.path.join('artifacts/plots',filename)
    plt.savefig(path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()
    training(config=parsed_args.config)