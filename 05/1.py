import tensorflow as tf


def inference(input_data):
    """
    正向传播计算
    :param input_data:
    :return:

    """
    input_shape = input_data.shape
    weights = tf.get_variable(name="weights",
                              shape=[input_shape[1], 1],
                              initializer=tf.initializers.random_normal)
    biases = tf.get_variable(name="biases",
                             shape=[1],
                             initializer=tf.initializers.constant)
    output = tf.matmul(input_data, weights) + biases
    return tf.nn.sigmoid(output)
