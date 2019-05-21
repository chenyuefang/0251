"""
1.梯度
2.负梯度
3.学习率
"""

def get_grad(theta, x, y):
    """
    求梯度
    """
    grad = 2 * (theta * x - y)
    return -grad


def get_cost(theta, x, y):
    return (theta * x - y) * 2


def gradient_descending(theta, x, y, learning_rate):
    theta = theta + get_grad(theta, x, y) * learning_rate
    return theta


y = 20
x = 1.1
theta = 0
learning_rate = 0.1
cost = get_grad(theta, x, y)
print(cost)
theta = gradient_descending(theta, x, y, learning_rate)
cost = get_cost(theta, x, y)
print(cost)
