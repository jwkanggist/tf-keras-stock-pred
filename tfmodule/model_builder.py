# -*- coding: utf-8 -*-
import tensorflow as tf
from model_config import model_chout_num
from train_config import TrainConfig

def get_model(input):

    trainconfig_worker = TrainConfig()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.SimpleRNN(model_chout_num['u1'], input_shape=(input, trainconfig_worker.train_input_size)))
    model.add(tf.keras.layers.Dense(1))

    return model

