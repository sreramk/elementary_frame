import cv2
import random

import tensorflow as tf

import numpy as np

from prepare_dataset.img_ds_manage import ImageDSManage


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


class SRNetworkManager:

    RMS_ERROR   = "RMS_ERROR"
    PSNR_ERROR  = "PSNR_ERROR"
    INPUT_LEARNER_MARGIN = "INPUT_LEARNER_MARGIN"

    def __init__(self, image_manager, filters_arch, bias_arch, strides_arch,
                 device, name, padding="SAME", param_path=None):
        """
        This class constructs the CNN model for training and testing.
        :param network_architecture: List of kernel sizes.
        :param load_parameters: parameter path, for restoring/saving variables
        :param image_manager: data-set manager
        """
        self.__filters_arch = filters_arch
        self.__bias_arch = bias_arch
        self.__strides_arch = strides_arch
        self.__device = device
        self.__image_manager = image_manager
        self.__padding = padding

        # Note, self.__bias_arch and self.__filters_arch must have the equal 0th dimension size

        self.__filters = []
        self.__biases = []
        self.__conv_layers = []
        self.__param_path = param_path
        self.__name = name

        self.__loss_network = {}

        with tf.device(self.__device):
            self.__network_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None],
                                                  name=name + "_input")
            self.__network_expected_out = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None],
                                                         name=name + "_expected_out")

        if self.__param_path is not None:
            self.__saver = tf.train.Saver()
        else:
            self.__saver = None

        self.__construct_network()

    def __construct_network(self):
        with tf.device(self.__device):
            if self.__param_path is None:
                for i in range(len(self.__filters_arch)):
                    self.__filters.append(tf.Variable(initial_value=tf.truncated_normal(shape=self.__filters_arch[i]),
                                                      name=self.__name + "_W_" + str(i)))
                    self.__biases.append(tf.Variable(initial_value=tf.truncated_normal(shape=self.__bias_arch[i]),
                                                     name=self.__name + "_B_" + str(i)))
            else:
                for i in range(len(self.__filters_arch)):
                    self.__filters.append(tf.get_variable(name=self.__name + "_W_" + str(i),
                                                          shape=self.__filters_arch[i]))
                    self.__biases.append(tf.get_variable(name=self.__name + "_B_" + str(i),
                                                         shape=self.__bias_arch[i]))

            cur_layer = tf.nn.conv2d(input=self.__network_input, filter=self.__filters[0],
                                     strides=self.__strides_arch[0], padding=self.__padding)
            cur_layer = tf.add(cur_layer, self.__biases[0])
            self.__conv_layers.append(cur_layer)

            for i in range(1, len(self.__filters)):
                cur_layer = tf.nn.conv2d(input=cur_layer, filter=self.__filters[i],
                                         strides=self.__strides_arch[i], padding=self.__padding)
                cur_layer = tf.add(cur_layer, self.__biases[i])
                self.__conv_layers.append(cur_layer)

    def restore_prams(self, session):
        if self.__param_path is not None:
            self.__saver.restore(session, self.__param_path)
        else:
            raise Exception("Error: parameter path not declared")

    def save_prams(self, session):
        if self.__param_path is not None:
            self.__saver.save(session, self.__param_path)
        else:
            raise Exception("Error: parameter path not declared")

    def get_network_output(self):
        return self.__conv_layers[len(self.__conv_layers)-1]

    #def set_rms_network_loss(self):



if __name__ == "__main__":
    for device in ['/gpu:0']:
        with tf.device(device):
            img_manager = ImageDSManage(["/home/sreramk/PycharmProjects/neuralwithtensorgpu/dataset/DIV2K_train_HR/"],
                                        image_buffer_limit=30)

            W_x = tf.Variable(initial_value=tf.truncated_normal(shape=[400, 400, 3]))

            w_1 = tf.Variable(initial_value=tf.truncated_normal(shape=[10, 10, 3, 10]))

            b_1 = tf.Variable(initial_value=tf.truncated_normal(shape=[10]))

            w_2 = tf.Variable(initial_value=tf.truncated_normal(shape=[2, 2, 10, 10]))

            b_2 = tf.Variable(initial_value=tf.truncated_normal(shape=[10]))

            w_3 = tf.Variable(initial_value=tf.truncated_normal(shape=[2, 2, 10, 2]))

            b_3 = tf.Variable(initial_value=tf.truncated_normal(shape=[2]))

            w_4 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 2, 2]))

            b_4 = tf.Variable(initial_value=tf.truncated_normal(shape=[2]))

            w_5 = tf.Variable(initial_value=tf.truncated_normal(shape=[10, 10, 10, 3]))

            b_5 = tf.Variable(initial_value=tf.truncated_normal(shape=[3]))

            input_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None], name="cnn_input")

            expected_output = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None], name="cnn_expected")

            # marks the image size
            min_x_t = tf.placeholder(dtype=tf.int32, shape=(), name="min_x_t")
            min_y_t = tf.placeholder(dtype=tf.int32, shape=(), name="min_y_t")

            # batch_size = tf.placeholder(dtype=tf.int32, shape=(), name="buffer_size")

            hidden1 = tf.nn.conv2d(input=input_data, filter=w_1, strides=[1, 1, 1, 1], padding="SAME")
            hidden1 = tf.add(hidden1, b_1)
            hidden1 = tf.nn.relu(hidden1)

            # hidden1 = tf.nn.max_pool(value=hidden1, ksize=5, strides=5, padding="VALID")
            # hidden1relu = tf.nn.relu(hidden1)
            # hidden1 = tf.divide(hidden1relu, tf.reduce_max(hidden1relu))
            # hidden1 = tf.reshape(tensor=hidden1, shape=[-1, min_y_t, min_x_t, 10 * 10, 3])
            # hidden1 = tf.reduce_mean(input_tensor=hidden1, axis=3)

            # hidden2 = hidden1

            hidden2 = tf.nn.conv2d(input=hidden1, filter=w_2, strides=[1, 1, 1, 1], padding="SAME")
            hidden2 = tf.add(hidden2, b_2)
            hidden2 = tf.nn.relu(hidden2)

            # hidden2 = tf.nn.max_pool(value=hidden2, ksize=5, strides=5, padding="VALID")

            hidden3 = tf.nn.conv2d(input=hidden2, filter=w_3, strides=[1, 1, 1, 1], padding="SAME")
            hidden3 = tf.add(hidden3, b_3)
            hidden3 = tf.nn.relu(hidden3)

            hidden4 = tf.nn.conv2d(input=hidden3, filter=w_4, strides=[1, 1, 1, 1], padding="SAME")
            hidden4 = tf.add(hidden4, b_4)
            hidden4 = tf.nn.relu(hidden4)

            hidden5 = tf.nn.conv2d(input=hidden2, filter=w_5, strides=[1, 1, 1, 1], padding="SAME")
            hidden5 = tf.add(hidden5, b_5)
            hidden5relu = tf.nn.relu(hidden5)
            hidden5 = tf.divide(hidden5relu, tf.reduce_max(hidden5relu))

            # network = hidden5
        with tf.device('/gpu:0'):
            network = hidden5

            # hidden2 = tf.abs(tf.sigmoid(hidden2))
            # hidden2 = tf.reshape(tensor=hidden2, shape=[-1, min_y_t, min_x_t, 10 * 10, 3])
            # hidden2 = tf.reduce_mean(input_tensor=hidden2, axis=3)

            prevent_learning_input = tf.abs(tf.subtract(expected_output, input_data))
            prevent_learning_input = tf.subtract(prevent_learning_input, tf.abs(tf.subtract(network, input_data)))
            prevent_learning_input = tf.reduce_mean(tf.square(prevent_learning_input))

            reduce_mean_loss = tf.reduce_mean(tf.square(tf.subtract(network, expected_output)))

            # reduce_mean_loss_in = tf.reduce_mean(tf.square(tf.subtract(network, input_data)))

            loss_c1 = log10(tf.divide(1, reduce_mean_loss))

            # loss_c2 = log10(tf.divide(1, prevent_learning_input))

            network_loss = tf.add(tf.multiply(-20.0,
                                              loss_c1),
                                  prevent_learning_input)  # tf.reduce_mean(tf.square(tf.subtract(network, expected_output)))

            # network_loss = tf.reduce_mean(tf.square(tf.subtract(W_x, expected_output)))

            """
                           - tf.reduce_mean(w_1) - tf.reduce_mean(w_2) - tf.reduce_mean(w_3) - \
                            tf.reduce_mean(w_4) - tf.reduce_mean(w_5)
            """
            # network_loss = tf.add(network_loss, reduce_mean_loss)
            # loss_c2 = tf.multiply(loss_c1, tf.multiply(-0.05, tf.add(network_loss, reduce_mean_loss_in)))

            # network_loss = tf.add(network_loss, loss_c2)

            # [w_1, b_1, w_2, b_2,w_3, b_3, w_4, b_4,w_5, b_5]
        with tf.device('/gpu:0'):
            adam_minimize = tf.train.AdamOptimizer().minimize(network_loss, var_list=[w_1, b_1, w_2, b_2, w_5, b_5])

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess2:

            init = tf.initialize_all_variables()

            sess.run(init)

            sess.graph.finalize()

            for j in range(100):

                print("Count: " + str(j))

                min_x, min_y, batch_down_sampled, batch_original = img_manager.get_batch(batch_size=6,
                                                                                         down_sample_factor=4,
                                                                                         min_x_f=70,
                                                                                         min_y_f=70)

                # for i in range(len(batch_original)):
                #    batch_original[i] = cv2.resize(batch_original[i], dsize=(52, 52))

                # batch_down_sampled = np.asarray(batch_down_sampled)

                # batch_down_sampled.fill(1.0)

                for i in range(10000):
                    # for j in range(10):

                    minimize, loss = sess.run(fetches=[adam_minimize, network_loss],
                                              feed_dict={input_data: batch_down_sampled,
                                                         expected_output: batch_original,
                                                         min_x_t: min_x,
                                                         min_y_t: min_y})
                    print("epoch :" + str(i))
                    print("loss: " + str(loss))

                for i in range(6):
                    # min_x, min_y, batch_down_sampled, batch_original = img_manager.get_batch(batch_size=1, down_sample_factor=10,
                    #                                                                     min_x_f=400, min_y_f=400 )

                    computed_image = sess.run(fetches=[network], feed_dict={input_data: batch_down_sampled,
                                                                            min_x_t: min_x,
                                                                            min_y_t: min_y})

                    print(len(computed_image))
                    print(len(computed_image[0]))
                    print(len(computed_image[0][0]))
                    print(len(computed_image[0][0][0]))
                    print(len(computed_image[0][0][0][0]))

                    cv2.imshow("original" + str(i) + "_" + str(j), batch_original[i])
                    cv2.imshow("down_sampled" + str(i) + "_" + str(j), batch_down_sampled[i])
                    cv2.imshow("computed" + str(i) + "_" + str(j), computed_image[0][i])
                cv2.waitKey(0)

                computed_image = None

                for i in range(5):
                    min_x, min_y, batch_down_sampled, batch_original = img_manager.get_batch(batch_size=1,
                                                                                             down_sample_factor=4,
                                                                                             min_x_f=400, min_y_f=400)

                    # batch_down_sampled = 1 - np.asarray(batch_down_sampled)

                    # if computed_image != None:
                    #    batch_down_sampled[0] = computed_image[0][0]

                    for x in range(10):

                        computed_image = sess.run(fetches=[network], feed_dict={input_data: batch_down_sampled,
                                                                            min_x_t: min_x,
                                                                            min_y_t: min_y})

                        batch_down_sampled[0] = computed_image


                    """
                    print (len(computed_image))
                    print (len(computed_image[0]))
                    print (len(computed_image[0][0]))
                    print (len(computed_image[0][0][0]))
                    print (len(computed_image[0][0][0][0]))
                    """

                    cv2.imshow("2original" + str(i) + "_" + str(j), batch_original[0])
                    cv2.imshow("2down_sampled" + str(i) + "_" + str(j), batch_down_sampled[0])
                    cv2.imshow("2computed" + str(i) + "_" + str(j), computed_image[0][0])

                cv2.waitKey(0)
