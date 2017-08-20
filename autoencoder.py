import cv2
import tensorflow as tf
import os
import numpy as np

dataset = []

for folder in os.listdir('./data'):
    folder_dataset = []
    for file in os.listdir('./data/{}'.format(folder)):
        file = './data/{}/{}'.format(folder, file)
        image = cv2.imread(file, 0)
        image = cv2.resize(image, (28, 28)).astype(np.float32)
        image = image / 255
        image = image.reshape(28 * 28 * 1)
        folder_dataset.append(image)
    dataset.append(folder_dataset)

dataset = np.array(dataset)
x_train = dataset[:, :250]
x_train = x_train.reshape(-1, 28 * 28 * 1)
x_test = dataset[:, 250:]
x_test = x_test.reshape(-1, 28 * 28 * 1)

input_dim = [None, 28 * 28 * 1]
layer_1_dim = 20 * 20 * 1
layer_2_dim = 20 * 20 * 1
layer_3_dim = 20 * 20 * 1
output_dim = 28 * 28 * 1

x_input = tf.placeholder(tf.float32, shape=input_dim)

layer_1_w = tf.get_variable('layer_1_w', shape=[28 * 28 * 1, 20 * 20 * 1], dtype=tf.float32)
layer_1_b = tf.get_variable('layer_1_b', shape=[20 * 20 * 1], dtype=tf.float32, initializer=tf.zeros_initializer())
layer_1 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(x_input, layer_1_w), layer_1_b))

layer_2_w = tf.get_variable('layer_2_w', shape=[20 * 20 * 1, 10 * 10 * 1], dtype=tf.float32)
layer_2_b = tf.get_variable('layer_2_b', shape=[10 * 10 * 1], dtype=tf.float32, initializer=tf.zeros_initializer())
layer_2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer_1, layer_2_w), layer_2_b))

layer_3_w = tf.get_variable('layer_3_w', shape=[10 * 10 * 1, 20 * 20 * 1], dtype=tf.float32)
layer_3_b = tf.get_variable('layer_3_b', shape=[20 * 20 * 1], dtype=tf.float32, initializer=tf.zeros_initializer())
layer_3 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(layer_2, layer_3_w), layer_3_b))

y_output_w = tf.get_variable('y_output_w', shape=[20 * 20 * 1, 28 * 28 * 1], dtype=tf.float32)
y_output_b = tf.get_variable('y_output_b', shape=[28 * 28 * 1], dtype=tf.float32, initializer=tf.zeros_initializer())
y_output = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer_3, y_output_w), y_output_b))

# layer_1 = tf.layers.dense(x_input, layer_1_dim, activation=tf.nn.tanh)
# layer_2 = tf.layers.dense(layer_1, layer_2_dim, activation=tf.nn.relu)
# layer_3 = tf.layers.dense(layer_2, layer_3_dim, activation=tf.nn.tanh)
# y_output = tf.layers.dense(layer_3, output_dim, activation=tf.nn.relu)

loss_mse = tf.reduce_mean(tf.square(x_input - y_output))
adam_optimizer = tf.train.AdamOptimizer().minimize(loss_mse)

init = tf.global_variables_initializer()

nb_epoch = 50
batch_size = 32
total_data = 2500
total_batch = int(total_data / batch_size)

import matplotlib.pyplot as plt

with tf.Session() as sess:
    sess.run(init)

    train_cost = []
    test_cost = []

    for e in xrange(nb_epoch):
        current = 0
        avg_train_cost = 0
        for i in xrange(total_batch):
            if current + batch_size < total_data:
                x = x_train[current:current + batch_size]
            else:
                x = x_train[current:]
            # print x
            current += batch_size
            _, cost = sess.run([adam_optimizer, loss_mse], feed_dict={x_input: x})
            avg_train_cost += cost

        avg_train_cost /= total_batch

        current = 0
        avg_test_cost = 0
        for i in xrange(total_batch):
            if current + batch_size < total_data:
                x = x_test[current:current + batch_size]
            else:
                x = x_test[current:]

            current += batch_size
            cost, output = sess.run([loss_mse, y_output], feed_dict={x_input: x})
            if i == 0 and (e == 0 or e == nb_epoch / 2 or e == nb_epoch - 1):
                cv2.imwrite(str(e) + '_' + str(i) + '_encode_.png', (x[23] * 255).reshape(28, 28).astype(np.uint8))
                cv2.imwrite(str(e) + '_' + str(i) + '_decode_.png', (output[23] * 255).reshape(28, 28).astype(np.uint8))
            avg_test_cost += cost

        avg_test_cost /= total_batch

        print 'epoch {}, training cost: {}'.format(e, avg_train_cost)
        print 'epoch {}, test_cost: {}'.format(e, avg_test_cost)
        train_cost.append(avg_train_cost)
        test_cost.append(avg_test_cost)
    plt.plot(train_cost)
    plt.plot(test_cost)
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()

