# -*- coding: utf-8 -*-
"""
Created on Sep 15 11:22:21 2018

@author: omerfarukkoc
"""
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

import time
from datetime import timedelta

# print(tf.__version__)

from tensorflow.examples.tutorials.mnist import input_data

#with tf.device('/gpu:0'):
mnist = input_data.read_data_sets("data/MNIST", one_hot=True, reshape=False)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_true = tf.placeholder(tf.float32, [None, 10])

filter1 = 16
filter2 = 32
filter3 = 256
out = 10

weight1 = tf.Variable(tf.truncated_normal([5, 5, 1, filter1], stddev=0.1))
bias1 = tf.Variable(tf.constant(0.1, shape=[filter1]))

weight2 = tf.Variable(tf.truncated_normal([5, 5, filter1, filter2], stddev=0.1))
bias2 = tf.Variable(tf.constant(0.1, shape=[filter2]))

weight3 = tf.Variable(tf.truncated_normal([7 * 7 * filter2, filter3], stddev=0.1))
bias3 = tf.Variable(tf.constant(0.1, shape=[filter3]))

weight4 = tf.Variable(tf.truncated_normal([filter3, out], stddev=0.1))
bias4 = tf.Variable(tf.constant(0.1, shape=[out]))

y1 = tf.nn.relu(tf.nn.conv2d(x, weight1, strides=[1, 1, 1, 1], padding='SAME') + bias1)
y1 = tf.nn.max_pool(y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

y2 = tf.nn.relu(tf.nn.conv2d(y1, weight2, strides=[1, 1, 1, 1], padding='SAME') + bias2)
y2 = tf.nn.max_pool(y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

flattened = tf.reshape(y2, shape=[-1, 7 * 7 * filter2])

y3 = tf.nn.relu(tf.matmul(flattened, weight3) + bias3)

logits = tf.matmul(y3, weight4) + bias4

y4 = tf.nn.softmax(logits)

x_ent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)

loss = tf.reduce_mean(x_ent)

correct_prediction = tf.equal(tf.argmax(y4, 1), tf.argmax(y_true, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

learningRate = 5e-4

optimize = tf.train.AdamOptimizer(learningRate).minimize(loss)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())

batch_size = 256

loss_graph = []


def training(iteration):
        for i in range(iteration):
            startTime = time.time()
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            feed_dict_train = {x: x_batch, y_true: y_batch}
            [_, train_loss] = sess.run([optimize, loss], feed_dict=feed_dict_train)

            loss_graph.append(train_loss)

            #print(i)
            if i % 100 == 0:
                train_acc = sess.run(accuracy, feed_dict=feed_dict_train)
                endTime = time.time()
                timeDifference = endTime-startTime

                print('Iter:', i, 'Accuracy:', train_acc, 'Loss:', train_loss, 'Time:', timedelta(seconds=timeDifference).total_seconds(), 'step/sec')



feed_dict_test = {x: mnist.test.images, y_true: mnist.test.labels}

def testingModel():
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    print('Testing Accuracy: ', acc)


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plot_example_errors():
    mnist.test.cls = np.argmax(mnist.test.labels, axis=1)
    y_pred_cls = tf.argmax(y4, 1)
    correct, cls_pred = sess.run([correct_prediction, y_pred_cls], feed_dict=feed_dict_test)
    incorrect = (correct == False)

    images = mnist.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = mnist.test.cls[incorrect]

    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

if __name__ == "__main__":
    training(1500)
    testingModel()
    #plot_example_errors()

    # plt.plot(loss_graph, '-k')
    # plt.title('Loss Graph')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.show()