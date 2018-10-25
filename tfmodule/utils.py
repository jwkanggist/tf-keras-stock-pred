import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def cal_mse(gt, pred):
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    return rmse

def cal_mae(gt, pred):
    mae = 100/len(gt) * np.sum((gt - pred)/gt)
    return mae

def show_as_plot(gt_y, pred_y,model_type,mse):
    plt.plot(gt_y, color='r', label='Ground Truth Y')
    plt.plot(pred_y, color='b', label='Prediction Y')
    plt.legend(loc=2)
    plt.ylabel('The closing value of Stock')
    plt.xlabel('Days')
    plt.title('%s based Stock prection of S&P500 data: MSE= %1.4f' % (model_type,mse))
    plt.show()