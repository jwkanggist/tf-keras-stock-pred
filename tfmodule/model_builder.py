# -*- coding: utf-8 -*-
import tensorflow as tf
from model_config import model_chout_num

def get_model(input):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.SimpleRNN(model_chout_num['u1'], input_shape=(input, 1)))
    model.add(tf.keras.layers.Dense(1))

    return model

