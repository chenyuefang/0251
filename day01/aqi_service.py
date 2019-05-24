"""
将AQI服务通过Flask发出至公网，供其它程序调用
"""

from flask import Flask
import read_data
import numpy as np

APP = Flask(__name__)  # 创建flask对象


@APP.route("/<name>")
def index(name):
    return "hello" + name


@APP.route("/aqi")
def get_aqi_value(input_data):
    """
    根据用户提供的输入数据，完成aqi值的预测
    """
    x = np.array(input_data)
    x = read_data.standard_data(x)
    with open('model.txt', 'r') as f:
        theta = np.array([float(line) for line in f.readlines()]).reshape(6, 1)
    return np.dot(x, theta)


if __name__ == "__main__":
    APP.run()  # 启动服务
