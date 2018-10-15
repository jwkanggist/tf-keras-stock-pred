# -*- coding: utf-8 -*-

class TrainConfig(object):
    def __init__(self):

        # the number of step between evaluation
        self.train_input_size = 1
        self.train_data_size  = 0.8
        self.test_data_size   = 0.8

        self.training_epochs  = 300

        self.optimizer       = 'adam'
        self.loss_fn         = 'mse'

        self.save_weight_name= 'save_weight_1.h5'

