#-*- coding: utf-8 -*-

import tensorflow as tf

a = tf.constant(10)
b = tf.constant(20)

with tf.Session() as sess:
    # run 을 하기 전에는 실제 연산이 되는 것이 아님
    c = a + b
    d = a * b
    print(c)
    print("addition with constants: %i" % sess.run(c))
    print("multiplication with constants: %i" % sess.run(d))


float32 = tf.float32
int16 = tf.int16
int32 = tf.int32
int64 = tf.int64

with tf.Session() as sess:
    print('float32.max:', float32.max, ', float32.min:', float32.min)
    print('int16.max:', int16.max, ', int16.min:', int16.min)
    print('int32.max:', int32.max, ', int32.min:', int32.min)
    print('int64.max:', int64.max, ', int64.min:', int64.min)


t = [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
t2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
t3 = tf.reshape(t2, [3, 3])

with tf.Session() as sess:
    print(t)
    print('tf.shape(t):', tf.shape(t))
    print('tf.size(t):', tf.size(t))
    print('tf.rank(t):', tf.rank(t))

    print('t2:', t2)
    print('t3:', t3)


t = [1, 2, 1, 3, 1, 1]
t_sq = tf.squeeze(t)

with tf.Session() as sess:
    print(t)
    print('tf.shape(t_sq):', tf.shape(t_sq))
    print(t_sq)


input = [[[1, 1, 1], [2, 2, 2]],
         [[3, 3, 3], [4, 4, 4]],
         [[5, 5, 5], [6, 6, 6]]]

with tf.Session() as sess:
    print('slice:', tf.slice(input, [1, 0, 0], [1, 1, 3]))


tensor = [[1, 2, 3], [4, 5, 6]]
tensor = tf.zeros_like(tensor)
with tf.Session() as sess:
    print(tensor)




