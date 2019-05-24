# -*- coding:utf-8 -*-
"""
读取数据
"""
import numpy as np
import pandas as pd


def read_aqi():
    aqi_data = pd.read_csv("aqi2.csv")  # 数据的读取
    cols = ['PM2.5', 'PM10', 'CO', 'No2', 'So2', 'O3']
    label = aqi_data["AQI"].values.reshape(-1, 1)  # reshape(-1, 1) 将数据转成列的形式
    x = aqi_data[cols].values  # 获取cols的值
    x = x.apply(lambda data: np.log(data), axis=0).values
    # x = (x - np.min(x) / (np.max(x) - np.min(x)))  # 固定公式，将数据标准化
    # 将数据拆分成3个部分，训练集占60%，验证集占30%，测试集占10%
    rows = len(x)
    train_rows = int(rows * 0.6)
    validation_rows = int(rows * 0.3)
    train_x, train_y = x[:train_rows], label[
                                       :train_rows]  # y = [0:5]  取0到4, 5 取不到    y = [  : 6]  取0到5    y = [4: ]  取4到最后，最后一个可以取到
    validation_x, validation_y = x[train_rows:train_rows + validation_rows], label[
                                                                             train_rows:train_rows + validation_rows]
    test_x, test_y = x[train_rows + validation_rows:], label[train_rows + validation_rows:]
    return (train_x, train_y), (validation_x, validation_y), (test_x, test_y)


def standard_data(input):
    """
    对输入数据进行标准化
    """
    return np.log(input)
