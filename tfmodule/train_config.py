# -*- coding: utf-8 -*-
import tensorflow as tf

class TrainConfig(object):
    def __init__(self):

        # the number of step between evaluation
        self.train_input_size = 1
        self.train_data_size  = 0.8
        self.test_data_size   = 0.8

        self.training_epochs  = 300

        self.optimizer       = 'adam'
        self.loss_fn         = 'mse'

        saved_model_folder_path = './saved_model'
        if not tf.gfile.Exists(saved_model_folder_path):
            tf.gfile.MakeDirs(saved_model_folder_path)
        self.save_weight_name= saved_model_folder_path + '/save_weight_1.h5'

