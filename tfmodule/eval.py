# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from data_loader import StockDataLoader
from data_loader import FileManager
from train_config import TrainConfig
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def cal_acc(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    return mse

def show(gt, pred):
    plt.plot(gt, color='r', label='Ground Truth')
    plt.plot(pred, color='b', label='Prediction')
    plt.legend(loc=2)
    plt.show()


def predict(dataloader,trainconfig_worker):
    X, y = dataloader.import_data(fm.filename)
    X_train,X_test = X[:int(X.shape[0] * trainconfig_worker.train_data_size)],X[int(X.shape[0] * trainconfig_worker.test_data_size):]
    y_train,y_test = y[:int(y.shape[0] * trainconfig_worker.train_data_size)],y[int(y.shape[0] * trainconfig_worker.test_data_size):]
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

    model = tf.keras.models.load_model(trainconfig_worker.save_weight_name)
    y_pred = model.predict(X_test)

    mse = cal_acc(y_test, y_pred)
    print("ACC : {0:.6f}".format(1-mse))

    return y_test, y_pred


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    trainconfig_worker = TrainConfig()
    fm = FileManager()

    dataloader = StockDataLoader()

    # model tranining
    gt, pred = predict(dataloader=dataloader, trainconfig_worker=trainconfig_worker)
    show(gt, pred)
