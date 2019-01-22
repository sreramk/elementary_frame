import tensorflow as tf

import numpy as np


kw_1 = tf.constant(value= [ [[ 1,  2], [ 3,  4], [ 5,  6]],
                            [[ 7,  8], [ 9, 10], [11, 12]],
                            [[13, 14], [15, 16], [17, 18]],
                            [[19, 20], [21, 22], [23, 24]]], shape=[4,3,2])


kw_2 = tf.constant(value= [ [[ 1,  2], [ 3,  4], [ 5,  6]],
                            [[ 7,  8], [ 9, 10], [11, 12]],
                            [[13, 14], [15, 16], [17, 18]],
                            [[19, 20], [21, 22], [23, 24]]], shape=[4,3,2])


w_1 = tf.Variable(initial_value=kw_1, name="w_1")

w_2 = tf.Variable(initial_value=kw_2, name="w_2")


w_3 = tf.tensordot(a=w_1, b=w_2, axes= [[], []], name="w_3")

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

    init = tf.initialize_all_variables()

    sess.run(init)
    result = sess.run(fetches=w_3)

    print("Result Size = " + str(len(result)) + " " +str(len(result[0]))+ " " + str(len(result[0][0]))+ " " +
                        str(len(result[0][0][0]))+ " " + str(len(result[0][0][0][0]))+ " " + str(len(result[0][0][0][0][0]))+ " "
                        );

    print ("Result = " + str(result));