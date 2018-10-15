# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from data_loader import StockDataLoader
from data_loader import FileManager
from train_config import TrainConfig
from model_builder import get_model
from model_config import model_config


def train(dataloader, trainconfig_worker):
    # load data
    X, y = dataloader.import_data(fm.filename, train=True)

    # set train, test data
    X_train, X_test = X[:int(X.shape[0] * trainconfig_worker.train_data_size)], X[int(X.shape[0] * trainconfig_worker.test_data_size):]
    y_train, y_test = y[:int(y.shape[0] * trainconfig_worker.train_data_size)], y[int(y.shape[0] * trainconfig_worker.test_data_size):]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], trainconfig_worker.train_input_size))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], trainconfig_worker.train_input_size))

    # get model
    model = get_model(dataloader.DAY_SIZE)
    model.compile(optimizer=model_config['optimizer'], loss=model_config['loss'])

    # learning
    history = model.fit(X_train, y_train, epochs=trainconfig_worker.training_epochs, validation_data=(X_test, y_test), shuffle=False)

    print("LOSS : {0:.6f}".format(history.history['loss'][-1]))
    print("VAL_LOSS : {0:.6f}".format(history.history['val_loss'][-1]))

    # save model weight
    tf.keras.models.save_model(model, trainconfig_worker.save_weight_name)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    trainconfig_worker = TrainConfig()
    fm = FileManager()

    dataloader = StockDataLoader()

    # model tranining
    train(dataloader=dataloader, trainconfig_worker=trainconfig_worker)
