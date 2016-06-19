#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

# tf graph input
X = tf.placeholder("float", [None, 3]) # x1, x2 and 1 (for bias)
Y = tf.placeholder("float", [None, 3]) # A, B, C ==> 3 classes

# set model weights
W = tf.Variable(tf.zeros([3, 3]))

# hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W))  # softmax

# learning rate 조절이 잘못되는 경우 테스트
learning_rate = 0.0001

# cost function
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# initializing the variables
init = tf.initialize_all_variables()

# launch the graph
with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

    print("===============================================")
    a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
    print(a, sess.run(tf.arg_max(a, 1)))

    b = sess.run(hypothesis, feed_dict={X:[[1, 3, 4]]})
    print(b, sess.run(tf.arg_max(b, 1)))

    c = sess.run(hypothesis, feed_dict={X:[[1, 1, 0]]})
    print(c, sess.run(tf.arg_max(c, 1)))

    all = sess.run(hypothesis, feed_dict={X:[[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
    print(all, sess.run(tf.arg_max(all, 1)))


"""
learning rate 을 조절해서 잘못된 성능이 나오는 경우를 테스트 해보자

learning_rate = 10.

0 nan [[-0.83333319  0.4166666   0.41666645]
 [ 1.66666687  2.91666746 -4.58333397]
 [ 1.66666627  4.16666698 -5.83333397]]
200 nan [[ nan  nan  nan]
 [ nan  nan  nan]
 [ nan  nan  nan]]

첫번째 이후의 값은 nan 으로 나오는 것을 확인할 수 있다. 학습이 일어나지 않고 있다.


learning_rate = 0.0001

0 1.09852 [[ -8.33333161e-06   4.16666580e-06   4.16666444e-06]
 [  1.66666687e-05   2.91666747e-05  -4.58333379e-05]
 [  1.66666614e-05   4.16666662e-05  -5.83333349e-05]]
200 1.08514 [[-0.00175495  0.00064726  0.00110769]
 [ 0.00293629  0.00489164 -0.00782793]
 [ 0.00294097  0.00740635 -0.01034732]]
400 1.07722 [[-0.0036236   0.00098153  0.00264207]
 [ 0.00522766  0.00818141 -0.01340907]
 [ 0.00524379  0.01319837 -0.01844216]]

학습은 일어나지만 cost 가 거의 줄지 않고 있다. 이런 경우 local minimum 에 빠질 수도 있다

"""
