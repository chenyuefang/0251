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
    return x.reshape(-1, 6), label.reshape(-1, 1)
