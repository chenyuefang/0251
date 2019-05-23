# -*- coding:utf-8 -*-
"""
读取数据
"""
import numpy as np
import pandas as pd


def read_aqi():
    aqi_data = pd.read_csv("aqi2.csv")
    cols = ['PM2.5', 'PM10', 'CO', 'No2', 'So2', 'O3']
    label = aqi_data["AQI"].values
    x = aqi_data[cols].values
    x = (x - np.min(x) / (np.max(x) - np.min(x)))
    # 将数据拆分成3个部分，训练集占60%，验证集占30%，测试集占10%
    rows = len(x)
    train_rows = int(rows * 0.6)
    validation_rows = int(rows * 0.3)
    train_x, train_y = x[:train_rows], label[:train_rows]
    validation_x, validation_y = x[train_rows:validation_rows], label[train_rows:validation_rows]
    test_x, test_y = x[validation_rows], label[validation_rows]
    return (train_x, train_y), (validation_x, validation_y), (test_x, test_y)
