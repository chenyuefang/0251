from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow as tf


def inference(input):
    weights = tf.get_variable(name="weights1",
                              shape=[784, 128],
                              initializer=tf.initializers.random_normal)
    biases = tf.get_variable(name="biases1",
                             shape=[128],
                             initializer=tf.initializers.constant)
    x = tf.matmul(input, weights) + biases
    x = tf.nn.sigmoid(x)
    weights2 = tf.get_variable(name="weights2",
                               shape=[128, 10],
                               initializer=tf.initializers.random_normal)
    biases2 = tf.get_variable(name="biases2",
                              shape=[10],
                              initializer=tf.initializers.constant)
    x2 = tf.matmul(x, weights2) + biases2
    x2 = tf.nn.sigmoid(x2)
    return x2


def train(input_x, input_y, ephocs=10, batch_size=1000):
    x = tf.placeholder("float", shape=[None, input_x.shape[1]], name="x-input")
    y = tf.placeholder("float", shape=[None, input_y.shape[1]], name="y-input")
    output = tf.nn.softmax(inference(x))
    # 定义损失函数
    cost = -tf.reduce_mean(y * tf.log(output))  # 逻辑回归的损失函数
    entropy_cost = tf.train.AdamOptimizer(0.031).minimize(cost)
    #  定义准确率的验证
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1)), "float"))
    batches = input_x.shape[0] // batch_size
    if input_x.shape[0] % batch_size != 0:
        batches += 1
    with tf.Session() as sess:
        tf.summary.FileWriter("logs,",sess.graph)
        sess.run(tf.global_variables_initializer())
        for _ in range(ephocs):
            for batch in range(batches):
                start = batch * batch_size % input_x.shape[0]
                end = min(start + batch_size, input_x.shape[0])
                sess.run([entropy_cost], feed_dict={x: input_x[start:end], y: input_y[start:end]})
            c = sess.run([cost, accuracy], feed_dict={x: input_x, y: input_y})
            print(c)


mnist = read_data_sets("data\\", one_hot=True)
train(mnist.train.images, mnist.train.labels)
# print(mnist.train.images[0])
# print(mnist.train.labels[0])
