# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import urllib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# data filename =====================================================
'''
    filename : ./input/all_stocks_5yr.csv
    stock : S&P 500
    duration : 5 years 
'''


class FileManager(object):

    def __init__(self):
        self.filename = 'all_stocks_5yr.csv'

class StockDataLoader(object):

    def __init__(self):

        self.SOURCE_URL     = 'https://github.com/CNuge/kaggle-code/raw/master/stock_data/'
        self.WORK_DIRECTORY = os.getcwd() + '/data'

        self.DAY_SIZE = 3
        self.file_path = ''


    def _processData(self, data):
        scl = MinMaxScaler()
        print(pd.__version__)

        # if pd.__version__ < '0.23':
        #     cl = data.reshape(data.shape[0],1)
        # elif pd.__version__ >= '0.23':

        data = data.as_matrix()
        data = data.reshape(data.shape[0],1)
        data = scl.fit_transform(data)

        X, Y = [], []
        for i in range(len(data)-self.DAY_SIZE-1):
            X.append(data[i:(i+self.DAY_SIZE), 0])
            Y.append(data[(i+self.DAY_SIZE), 0])
        return np.array(X), np.array(Y)


    # function module for data set loading  ============================
    def _download(self, filename):

        if not tf.gfile.Exists(self.WORK_DIRECTORY):
            tf.gfile.MakeDirs(self.WORK_DIRECTORY)
            tf.logging.info(" %s is not exist" % self.WORK_DIRECTORY)

        filepath = os.path.join(self.WORK_DIRECTORY, filename)

        tf.logging.info('filepath = %s' % filepath)
        tf.logging.info(self.SOURCE_URL+ filename+"/4")

        if not tf.gfile.Exists(filepath):
            filepath, _ = urllib.request.urlretrieve(self.SOURCE_URL+ filename, filepath)
            with tf.gfile.GFile(filepath) as f:
                size = f.size()
                tf.logging.info('Successfully downloaded', filename, size, 'bytes.')

            tf.logging.info('[download_stock_dataset] filepath = %s' % filepath)

        return filepath


    def import_data(self,inputfilename):

        tf.logging.info('[Input_fn] download if the files does not exist')

        self.file_path = self._download(filename=inputfilename)
        data = pd.read_csv(self.file_path)
        data = data[data['Name']=='MMM'].close
        data_numpy, label_numpy = self._processData(data)

        return data_numpy, label_numpy
