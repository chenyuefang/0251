"""
1.梯度
2.负梯度
3.学习率
"""


def get_grad(theta, x, y):
    """
    求梯度
    """
    grad = 2 * (theta * x - y) * x
    return -grad


def get_cost(theta, x, y):
    """
    学习率
    """
    return (theta * x - y) ** 2


def gradient_descending(theta, x, y, learning_rate):
    """
   梯度下降
    """
    for _ in range(20):
        theta = theta + get_grad(theta, x, y) * learning_rate
        print(get_cost(theta,x,y))


y = 20
x = 1.1
theta = 0
learning_rate = 0.1
theta = gradient_descending(theta, x, y, learning_rate)
