# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from data_loader import StockDataLoader
from data_loader import FileManager
from train_config import TrainConfig
import utils

def predict(dataloader,trainconfig_worker):
    X, y = dataloader.import_data(fm.filename, train=True)
    _X, _y = dataloader.import_data(fm.filename, train=False)
    X_train,X_test = X[:int(X.shape[0] * trainconfig_worker.train_data_size)],X[int(X.shape[0] * trainconfig_worker.test_data_size):]
    y_train,y_test = y[:int(y.shape[0] * trainconfig_worker.train_data_size)],y[int(y.shape[0] * trainconfig_worker.test_data_size):]
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],trainconfig_worker.train_input_size))

    model = tf.keras.models.load_model(trainconfig_worker.save_weight_name)
    y_pred = model.predict(X_test)

    y_test = y_test * _y[int(X.shape[0] * trainconfig_worker.test_data_size):]
    y_pred = y_pred[:,0] * _y[int(X.shape[0] * trainconfig_worker.test_data_size):]

    mse = utils.cal_mse(y_test, y_pred)
    mae = utils.cal_mae(y_test, y_pred)
    print("ACC MSE : {0:.6f}".format(mse))
    print("ACC MAE : {}".format(mae))

    return y_test, y_pred


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    trainconfig_worker = TrainConfig()
    fm = FileManager()

    dataloader = StockDataLoader()

    # model tranining
    gt, pred = predict(dataloader=dataloader, trainconfig_worker=trainconfig_worker)
    utils.show(gt, pred)
