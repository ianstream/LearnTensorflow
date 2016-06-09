import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# 오퍼레이션을 정의, 실제 결과는 run 시에 결정됨
add = tf.add(a, b)
mul = tf.mul(a, b)

with tf.Session() as sess:
    print("addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))
