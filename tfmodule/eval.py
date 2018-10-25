# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from data_loader import StockDataLoader
from data_loader import FileManager
from train_config import TrainConfig
from model_config import model_config
import utils

def predict(dataloader,trainconfig_worker):
    X, y = dataloader.import_data(fm.filename, train=True)
    _X, _y = dataloader.import_data(fm.filename, train=False)

    X_train,X_test = X[:int(X.shape[0] * trainconfig_worker.train_data_size)],\
                     X[int(X.shape[0] * trainconfig_worker.test_data_size):]

    y_train,y_test = y[:int(y.shape[0] * trainconfig_worker.train_data_size)],\
                     y[int(y.shape[0] * trainconfig_worker.test_data_size):]

    X_test = X_test.reshape((X_test.shape[0],
                             X_test.shape[1],
                             trainconfig_worker.train_input_size))

    model = tf.keras.models.load_model(trainconfig_worker.save_weight_name)
    y_pred = model.predict(X_test)

    y_test = y_test * _y[int(X.shape[0] * trainconfig_worker.test_data_size):]
    y_pred = y_pred[:,0] * _y[int(X.shape[0] * trainconfig_worker.test_data_size):]

    mse = utils.cal_mse(y_test, y_pred)
    mae = utils.cal_mae(y_test, y_pred)
    print("ACC RMSE : {0:.6f}".format(mse))
    print("ACC MAE : {}".format(mae))

    return X_test,y_test, y_pred,mse


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    trainconfig_worker = TrainConfig()
    fm = FileManager()

    dataloader = StockDataLoader()

    # model tranining
    input_x, gt_y, pred_y,mse = predict(dataloader=dataloader, trainconfig_worker=trainconfig_worker)
    utils.show_as_plot(gt_y, pred_y,
                       model_type=model_config['model_type'],
                       mse = mse)
