import pandas as pd
import numpy as np


def read_data():
    data = pd.read_csv("wdbc.csv")
    # 对数据进行shuffle算法进行乱序排列
    data = data.simple(frac=1.0)
    cols = []
