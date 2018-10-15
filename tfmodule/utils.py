import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def cal_mse(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    return mse

def cal_mae(gt, pred):
    mae = 100/len(gt) * np.sum((gt - pred)/gt)
    return mae

def show(gt, pred):
    plt.plot(gt, color='r', label='Ground Truth')
    plt.plot(pred, color='b', label='Prediction')
    plt.legend(loc=2)
    plt.show()