import random

import tensorflow as tf

import numpy as np

def generate_training_data(size, batches):

    result = []

    for i in range(batches):

        tempIn  = []

        tempOut = []

        for j in range(size):
            x1 = random.random()
            x2 = random.random()
            x3 = random.random()
            tempIn.append( [[x1, x2, x3]] )
            tempOut.append( [[x1+x2+x3]] )


        result.append((tempIn, tempOut))


    return result

def generate_network(batchSize):

    if batchSize is None:
        batchSize = 1

    w_multiple = tf.constant(value=1.0, shape=[batchSize, 3, 3], dtype=tf.float32)

    b_multiple = tf.constant(value=1.0, shape=[batchSize, 1, 3], dtype=tf.float32)


    we_multiple = tf.constant(value=1.0, shape=[batchSize, 1, 3], dtype=tf.float32)

    be_multiple = tf.constant(value=1.0, shape=[batchSize, 1, 1], dtype=tf.float32)

    input_layer = tf.placeholder(dtype=tf.float32, shape= [batchSize,1,3], name= "input_layer")

    input_layer_base = tf.placeholder(dtype=tf.float32, shape= [1,1,3], name= "input_layer_base")


    w_1 = tf.Variable(initial_value=[tf.truncated_normal(shape=[3,3])], name= "w_1")

    xw_1 = tf.multiply(w_1, w_multiple)

    print(xw_1)

    b_1 = tf.Variable(initial_value=[tf.truncated_normal(shape=[1,3])], name= "b_1")

    xb_1 = tf.multiply(b_1, b_multiple)

    w_2 = tf.Variable(initial_value=[tf.truncated_normal(shape=[3,3])], name= "w_2")

    xw_2 = tf.multiply(w_2, w_multiple)

    b_2 = tf.Variable(initial_value=[tf.truncated_normal(shape=[1,3])], name= "b_2")

    xb_2 = tf.multiply(b_2, b_multiple)

    w_3 = tf.Variable(initial_value=[tf.truncated_normal(shape=[1,3])], name= "w_3")

    xw_3 = tf.multiply(w_3, we_multiple)

    b_3 = tf.Variable(initial_value=[tf.truncated_normal(shape=[1,1])], name= "b_3")

    xb_3 = tf.multiply(b_3, be_multiple)

    network = tf.matmul (input_layer, xw_1)
    network = tf.add(network, xb_1)
    network = tf.multiply(3.0, tf.sigmoid(network))

    network = tf.matmul (network, xw_2)
    network = tf.add(network, xb_2)
    network = tf.multiply(3.0, tf.sigmoid(network))

    network = tf.matmul (network, tf.transpose(xw_3, perm=[0,2,1]))
    network = tf.add(network, xb_3)
    network = tf.multiply(3.0, tf.sigmoid(network))

    network_base = tf.matmul(input_layer_base, w_1)
    network_base = tf.add(network_base, b_1)
    network_base = tf.multiply(3.0, tf.sigmoid(network_base))

    network_base = tf.matmul(network_base, w_2)
    network_base = tf.add(network_base, b_2)
    network_base = tf.multiply(3.0, tf.sigmoid(network_base))

    network_base = tf.matmul(network_base, tf.transpose(w_3, perm=[0, 2, 1]))
    network_base = tf.add(network_base, b_3)
    network_base = tf.multiply(3.0, tf.sigmoid(network_base))
    #tf.summary.scalar("xw_3", xw_3)
    return  w_1, w_2, w_3, b_1, b_2, b_3, network, network_base, input_layer, input_layer_base


expected_output = tf.placeholder(dtype= tf.float32, shape= [1000,1,1], name= "expected_output")

expected_output_base = tf.placeholder(dtype= tf.float32, shape= [1,1,1])


w_1, w_2, w_3, b_1, b_2, b_3, network, network_base, input_layer, input_layer_base = generate_network(1000)

network_cost = tf.reshape(tensor= tf.reduce_mean( input_tensor= tf.square(tf.subtract(network, expected_output) ),
                               axis= 0), shape= [])


network_cost_base = tf.square(tf.subtract(network_base, expected_output_base) )

opt1 = tf.train.AdamOptimizer().minimize(network_cost, var_list=[w_1, b_1])

opt2 = tf.train.AdamOptimizer().minimize(network_cost, var_list=[w_2, b_2])

opt3 = tf.train.AdamOptimizer().minimize(network_cost, var_list=[w_3, b_3])

#opt4 = tf.train.AdamOptimizer().minimize(network_cost, var_list=[w_1, b_1,w_2,b_2,w_3,b_3])

# Create a summary to monitor cost tensor
network_cost_scalar = tf.summary.scalar(name= "network_cost_scalar", tensor= network_cost)
# Create a summary to monitor accuracy tensor
#tf.summary.scalar("network_cost_base", network_cost_base)
# Merge all summaries into a single op
#merged_summary_op = tf.summary.merge_all()

print (network_cost_scalar)

# Creates a session with log_device_placement set to True.
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

    train_data = generate_training_data(1000, 100)

    testing_data = generate_training_data(1, 1000)

    init = tf.initialize_all_variables()

    sess.run(init)

    summary_writer = tf.summary.FileWriter("/home/sreramk/PycharmProjects/neuralwithtensorgpu/visual/", graph=sess.graph)

    # Runs the op.

    for i in range(len(train_data)):

        if  i < (100 / 3 ):
            __, summary = sess.run(fetches=[opt1, network_cost_scalar], feed_dict={input_layer: train_data[i][0],
                                                                                   expected_output: train_data[i][1]})
        elif i < (2*(100) / 3):
            __, summary = sess.run(fetches=[opt2, network_cost_scalar], feed_dict={input_layer: train_data[i][0],
                                                                                   expected_output: train_data[i][1]})
        else:
            __, summary = sess.run(fetches=[opt3, network_cost_scalar], feed_dict={input_layer: train_data[i][0],
                                                                                   expected_output: train_data[i][1]})
        summary_writer.add_summary(summary, i)

        print ("epoch: " + str(i))


        #print ("w_1 = " + str(sess.run(fetches=w_1, feed_dict={input_layer : train_data[i][0],
        #                                             expected_output : train_data[i][1] })))

    result = []

    for i in range(len(testing_data)):
        res = sess.run(fetches=network_base, feed_dict={input_layer_base: testing_data[i][0],
                                                             expected_output_base: testing_data[i][1]})
        res2 = sess.run(fetches=network_cost_base, feed_dict={input_layer_base: testing_data[i][0],
                                                        expected_output_base: testing_data[i][1]})

        print("")
        print("input = " + str(testing_data[i][0]))
        print("expected output = " + str(testing_data[i][1]))
        print("network's output= " + str(res))
        print("error = " + str(res2))
        result.append(res2)


    print ("average error = " + str(np.average(res2, axis=1)))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/home/sreramk/PycharmProjects/neuralwithtensorgpu/visual/ " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")


    #tf.train.GradientDescentOptimizer