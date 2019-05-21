"""
1.梯度
2.负梯度
3.学习率
"""
import numpy as np


def get_grad(theta, x, y):
    """
    求梯度
    """
    grad = 2 * (theta * x - y) * x
    return -grad


def get_cost(theta, x, y):
    """
    学习率
     return (theta * x - y) ** 2  单个X
     X:是一个矩阵
     Y:是一个矩阵
     Z:是一个矩阵
    """
    return np.sum((np.dot(x, theta) - y) ** 2)


def gradient_descending(theta, x, y, learning_rate):
    """
   梯度下降
    """
    for _ in range(20):
        theta = theta + get_grad(theta, x, y) * learning_rate
        print(get_cost(theta, x, y))


x = np.array([[1, 2], [1, 2]])
y = np.array([0, 0])
theta = np.array([1, 1])

print(get_cost(theta, x, y))
