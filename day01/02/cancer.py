import numpy as np
import data_reader


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def get_cost(theta, x, y):
    h = sigmoid(np.dot(x, theta))
    cost = np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))


def get_grad(theta, x, y):
    h = sigmoid(np.dot(x, theta))
    return np.dot(np.transpose(x), h - y)


def gradient_descending(theta, x, y, learning_rate):
    for _ in range(20):
        theta = theta - learning_rate * get_grad(theta, x, y)
        cost = get_cost(theta, x, y)
        print (cost)


x, y = data_reader.read_data()
theta = np.zeros(x.shape[1], 1)
learning_rate = 0.001
gradient_descending(theta, x, y, learning_rate)
