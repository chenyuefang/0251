import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


def inference(input):
    weights = tf.get_variable(name="weights",
                              shape=[784, 10],
                              initializer=tf.initializers.random_normal)
    biases = tf.get_variable(name="biases",
                             shape=[10],
                             initializer=tf.initializers.constant)
    x = tf.matmul(input, weights) + biases
    return tf.nn.sigmoid(x)


def train(input_x, input_y, ephocs=10, batch_size=100):
    x = tf.placeholder("float", shape=[None, input_x.shape[1]], name="x-input")
    y = tf.placeholder("float", shape=[None, input_y.shape[1]], name="y-input")
    output = tf.nn.softmax(inference(x))
    # 定义损失函数
    cost = -tf.reduce_sum(y * tf.log(output) + (1 - y) * tf.log(1 - output))
    entropy_cost = tf.train.GradientDescentOptimizer(0.0003).minimize(cost)
    batches = input_x.shape[0] // batch_size
    if input_x.shape[0] % batch_size != 0:
        batches += 1

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for _ in range(ephocs):
            for batch in range(batches):
                start = batch * batch_size % input_x.shape[0]
                end = min(start + batch_size, input_x.shape[0])
                sess.run([entropy_cost], feed_dict={x: input_x[start:end], y: input_y[start:end]})
            print(sess.run([cost], feed_dict={x: input_x, y: input_y}))


mnist = read_data_sets("data\\", one_hot=True)
train(mnist.train.images, mnist.train.labels)
