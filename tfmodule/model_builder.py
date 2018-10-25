# -*- coding: utf-8 -*-
import tensorflow as tf
from model_config import model_chout_num
from train_config import TrainConfig

def get_model(input,model_type='lstm'):

    trainconfig_worker = TrainConfig()
    model = tf.keras.models.Sequential()

    if model_type == 'lstm':
        ## for use of LSTM
        model.add(
            tf.keras.layers.LSTM(units=model_chout_num['u1'],
                                 input_shape=(input,trainconfig_worker.train_input_size)))
    elif model_type == 'gru':
        model.add(
            tf.keras.layers.GRU(units=model_chout_num['u1'],
                                input_shape=(input,trainconfig_worker.train_input_size)))
    else:
        model.add(
            tf.keras.layers.SimpleRNN(units=model_chout_num['u1'],
                                      input_shape=(input, trainconfig_worker.train_input_size)))

    model.add(tf.keras.layers.Dense(1))

    return model

