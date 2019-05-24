"""
是为了演示梯度下降，梯度下降有几个重要因素：
1.梯度
2.负梯度
3.学习率
"""
import numpy as np
import read_data
from show import show_cost


def get_grad(theat, x, y):
    """
    :param theat:矩阵
    :param x: 矩阵
    :param y: 矩阵
    :return: 矩阵
    """
    grad = np.dot(np.transpose(x), (np.dot(x, theat) - y))
    # np.dot 矩阵的乘法
    return grad


def gradient_descending(theta, x, y, vx, vy, learning_rate):
    train_costs = []
    # 记录训练过程中的产生的cost
    validation_costs = []
    # 记录验证集上产生的cost
    for i in range(20):
        theta = theta - get_grad(theta, x, y) * learning_rate
        train_costs.append(get_cost(theta, x, y))
        validation_costs.append(get_cost(theta, vx, vy))
    show_cost(train_costs, validation_costs)
    # print(get_cost(theta,x,y))


def get_cost(theta, x, y):
    """
    :param theta: 矩阵
    :param x: 矩阵
    :param y: 矩阵
    :return:
    """
    return np.mean((np.dot(x, theta) - y) ** 2)  # 取均值


def get_aqi_value(input_data):
    """
    根据用户提供的输入数据，完成aqi值的预测
    :param input_data:
    :return:
    """
    x = np.array(input_data)
    return np.dot(x, theta)


train_data, validation_data, test_data = read_data.read_aqi()

theta = np.zeros((6, 1))  # 六行一列的零
learning_rate = 0.00000011
# print(theta)
gradient_descending(theta, train_data[0], train_data[1], validation_data[0], validation_data[1], learning_rate)
# test_model (theta , test_data[0] ,test_data[1]  )
aqi_value = get_aqi_value([33, 56, 7, 27, 0, 82, 101])
print(aqi_value)
