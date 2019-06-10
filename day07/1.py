import tensorflow as tf
import input_data
import cifar_inference
import os
import numpy as np

# 常量定义
BATCH_SIZE = 100
REGULARAZTION_RATE = 0.0001
LEARNING_RATE_BASE = 0.0009
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 10
MOVING_AVERAGE_DECAY = 0.99  # 衰减速率

# 定义模型保存的路径和文件名
MODEL_SAVE_PATH = "models"
MODEL_NAME = "model.ckpt"
import tensorflow as tf
import cifar_inference
import cifar_train
import input_data
import numpy as np

IS_MODEL_INIT = False
SESS = None


def get_weight(shape, std=0.1, regularization=None):
    """
    根据用户输入的矩阵信息，创建匹配的权重矩阵
    :param shape: 权重大小矩阵，该参数为一个list()，如[3,3,1,16]
    :param regularization:对权重矩阵进行正则化的参数
    :return:创建成功的矩阵信息
    """
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=std))
    if regularization:
        tf.add_to_collection("losses", regularization(weights))
    return weights


def get_biases(shape):
    """
    根据用户传入的矩阵大小创建偏置信息
    :param shape: 用户传入的矩阵大小，为一个list()，如[3]
    :return: 与shape大小匹配的矩阵信息
    """
    biases = tf.get_variable("biases",
                             shape,
                             initializer=tf.constant_initializer(0))
    return biases


def convolution(x, w, strides):
    """
    卷积运算
    :param x:卷积输入
    :param w:卷积filter
    :param strides:滑动窗步长
    :return:卷积之后的结果
    """
    return tf.nn.conv2d(x, w, strides=strides, padding="SAME")


def pooling(input, ksize, strides):
    """
    :param input:池化输入
    :param ksize:池化的滑动窗口大小
    :param strides:滑动大小
    :return:
    """
    return tf.nn.max_pool(input, ksize, strides=strides, padding="VALID")


def avg_pooling(input, ksize, strides):
    """
    使用avg_pooling进行池化
    :param input: 池化输入
    :param ksize: 滑动窗口大小
    :param strides: 滑动大小
    :return:
    """
    return tf.nn.avg_pool(input, ksize, strides=strides, padding='VALID')


def full_connection_nn(input_tensor, input_size, output_size, regularization=None):
    """
    全连接层运算
    :param input_tensor:输入的tensor
    :param input_size:输入tensor的规模
    :param output_size:输出tensor的规模
    :param regularization:对权重进行正则化的函数
    :return:
    """
    weights = get_weight([input_size, output_size], 0.1, regularization)
    biases = get_biases(shape=[output_size])
    result = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
    return result


def inference(input_tensor, regularization):
    """
    正向传播计算过程
    :param input_tensor:输入tensor
    :param regularization:全局正则化函数
    :return:
    """
    input_tensor = tf.reshape(input_tensor, shape=[-1, 32, 32, 3])

    # 卷积层1
    with tf.variable_scope("conv1"):
        w_conv1 = get_weight([5, 5, 3, 32], 0.0001, regularization)
        b_conv1 = get_biases([32])
        conv1 = tf.nn.selu(convolution(input_tensor, w_conv1, strides=[1, 1, 1, 1]) + b_conv1)
        max_pool = pooling(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1])
    # 卷积层2
    with tf.variable_scope("conv2"):
        w_conv2 = get_weight(shape=[5, 5, 32, 32], std=0.01, regularization=regularization)
        b_conv2 = get_biases(shape=[32])
        conv2 = tf.nn.selu(convolution(max_pool, w_conv2, strides=[1, 1, 1, 1]) + b_conv2)
        max_pool2 = pooling(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1])

    # 卷积层3
    with tf.variable_scope("conv3"):
        w_conv3 = get_weight([3, 3, 32, 64], 0.01, regularization=regularization)
        b_conv3 = get_biases(shape=[64])
        conv3 = tf.nn.selu(convolution(max_pool2, w_conv3, [1, 1, 1, 1]) + b_conv3)
        max_pool3 = pooling(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1])

    # 第一层大小为[3136, 1024]
    with tf.variable_scope("fc1"):
        max_pool3 = tf.reshape(max_pool3, [-1, 576])
        fc1 = full_connection_nn(max_pool3, 576, 10,
                                 regularization=regularization)
        fc1 = tf.nn.relu(fc1)
        return fc1


def train(images, labels, test_iamges, test_labels):
    """
    训练模型
    :param images:为训练模型提供的图片信息
    :param labels:训练数据对应的标签信息
    :return:
    """
    # 32*32*3
    x = tf.placeholder("float32", shape=[None, 3072])
    y = tf.placeholder("float32", shape=[None, 10])

    regularization = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    output_labels = cifar_inference.inference(x, regularization)

    global_step = tf.Variable(0, trainable=False)
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_average.apply(tf.trainable_variables())
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y, 1), tf.arg_max(output_labels, 1)), "float"))
    # 计算学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               50000 // BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        loop_time = 50000 // BATCH_SIZE if 50000 % BATCH_SIZE == 0 else 50000 // BATCH_SIZE + 1
        try:
            for i in range(TRAINING_STEPS):
                if coord.should_stop():
                    break
                loss_value = 0.0
                for t in range(loop_time):
                    start = t * BATCH_SIZE % 50000
                    end = min(start + BATCH_SIZE, 50000)
                    _, loss_value, step = sess.run([train_op, loss, global_step],
                                                   feed_dict={x: images[start:end], y: labels[start:end]})
                acc = sess.run(accuracy, feed_dict={x: test_iamges, y: test_labels})
                # 每轮保存一次模型
                print("经过 %d 轮训练, 整体损失为： %g， 准确率为：%g." % (i + 1, loss_value, acc))
                if acc > 0.7:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        except tf.errors.OutOfRangeError:
            print("Done training--epoch limit reached.")
        finally:
            coord.request_stop()
    coord.join(threads)


def main(argv=None):
    (images, labels), (
        t_images, t_labels) = input_data.load_data()  # input_data.distorted_inputs("../data/cifar/", 128)
    images = np.reshape(images, (50000, 3072))
    t_images = np.reshape(t_images, (10000, 3072))
    tmp = []
    tmp_t = []
    for i in range(0, 50000):
        data = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.int)
        data[labels[i]] = 1
        tmp.append(data)
        del data
    del labels
    for i in range(0, 10000):
        data = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.int)
        data[t_labels[i]] = 1
        tmp_t.append(data)
        del data
    del t_labels
    train(images, np.array(tmp), t_images, tmp_t)


if __name__ == "__main__":
    tf.app.run()


def evaluate(images):
    """
    对用户输入的图片进行分类，并返回分类结果
    :param images: 用户输入的图片信息
    :return: 用户输入图片的分类结果，为list
    """
    with tf.Graph().as_default() as g:
        x = tf.placeholder("float32", shape=[None, 3072], name="x-input")
        validate_feed = {x: images}
        regularization = tf.contrib.layers.l2_regularizer(cifar_train.REGULARAZTION_RATE)
        output_labels = tf.nn.softmax(cifar_inference.inference(x, regularization))

        variable_average = tf.train.ExponentialMovingAverage(cifar_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        global SESS
        global IS_MODEL_INIT
        if not IS_MODEL_INIT:
            SESS = tf.Session()
            ckpt = tf.train.get_checkpoint_state(cifar_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(SESS, ckpt.model_checkpoint_path)
            IS_MODEL_INIT = True
        try:
            predicted = SESS.run(tf.argmax(output_labels, 1), feed_dict=validate_feed)
            return predicted
        finally:
            SESS.close()


def read_images(image_folder):
    """
    遍历文件夹下所有图片，并将图片放入到Images中返回
    :param image_folder: 存放图片的文件夹
    :return: 图片数据，期望序列
    """
    # 读取图片信息
    images = []
    excepted = []
    for root, _, files in os.walk(image_folder):
        for f in files:
            excepted.append(int(f.split(".")[0]))
            arr = []
            img = Image.open(os.path.join(root, f)).convert("RGB")
            if img.width != 32 or img.height != 32:
                img = img.resize((32, 32))
            for i in range(32):
                for j in range(32):
                    rgb = img.getpixel((j, i))
                    r = 1.0 - rgb[0] / 255.0
                    g = 1.0 - rgb[1] / 255.0
                    b = 1.0 - rgb[2] / 255.0
                    arr.append(r)
                    arr.append(g)
                    arr.append(b)
            images.append(arr)
    return np.array(images), excepted


if __name__ == "__main__":
    compents = {}
    compents[0] = "air plane"
    compents[1] = "automobile"
    compents[2] = "bird"
    compents[3] = "cat"
    compents[4] = "deer"
    compents[5] = "dog"
    compents[6] = "frog"
    compents[7] = "horse"
    compents[8] = "ship"
    compents[9] = "truck"
    image_folder = "test_images"
    images, excepted = read_images(image_folder)
    predicted = evaluate(images)
    print("\n\n\n---------------------------------------------------------------------")
    print("%16s\t%16s\t%10s" % ("Excepted", "Actually", "Is Right"))
    print("---------------------------------------------------------------------")
    for i in range(len(excepted)):
        print("%16s\t%16s\t%4s" % (
            compents[excepted[i]], compents[predicted[i]],
            "Y" if compents[excepted[i]] == compents[predicted[i]] else "N"))
