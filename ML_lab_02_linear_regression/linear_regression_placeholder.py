#-*- coding: utf-8 -*-

import tensorflow as tf


x_data = [1, 2, 3]
y_data = [1, 2, 3]

"""
try to find vlaues for W and b compute y_data = W * x_data + b
(we know that W should be 1 and b 0, but Tensorflow will figure that out for us)
"""
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# our hypothesis
hypothesis = W * X + b

# simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# before starting, initilize the variables. we will 'run' this first
init = tf.initialize_all_variables()

# launch the graph
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(2001):
    # 프로그램이 실행되는 시점에 값이 지정됨
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))

# placeholder 의 사용으로 모델의 재작성 없이 원하는 값을 넘겨줘서 구할 수 있게 됨
print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))

