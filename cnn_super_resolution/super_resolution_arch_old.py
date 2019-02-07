# Copyright (c) 2019 K Sreram, All rights reserved

import cv2
import os

import tensorflow as tf

from prepare_dataset.img_ds_manage import ImageDSManage


from datetime import datetime



def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


class InvalidOutputLayer(Exception):
    """
    Raised when the layer in the network requested as the output layer is invalid.
    """
    pass


class SRNetworkManager:
    RMS_ERROR = "RMS_ERROR"
    PSNR_ERROR = "PSNR_ERROR"
    TRANSFORM_RESTRAIN_ERROR = "INPUT_LEARNER_MARGIN"

    def __init__(self, name, param_path,
                 filters_arch=None, bias_arch=None, strides_arch=None,
                 device=None, padding="SAME"):
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
        self.__padding = padding

        # Note, self.__bias_arch and self.__filters_arch must have the equal 0th dimension size

        self.__filters = []
        self.__biases = []
        self.__conv_layers = []
        self.__param_path = param_path
        self.__name = name

        self.__loss_network = {}

        self.__optimizer = None

        self.__output_layers = {}

        with tf.device(self.__device):
            self.__network_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None],
                                                  name=self.__name + "_input")
            self.__network_expected_out = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None],
                                                         name=self.__name + "_expected_out")

        self.__saver = None

    def number_of_layers(self):
        return len(self.__conv_layers)

    def get_layer(self, index):
        if index > self.number_of_layers():
            return None
        return self.__conv_layers[index]

    def initialize_saver(self):
        self.__saver = tf.train.Saver()

    def is_filter_construct_ready(self):
        if (self.__filters_arch is not None) and \
                (self.__bias_arch is not None) and \
                (self.__strides_arch is not None) and \
                (self.__device is not None):
            return True
        return False

    def is_layer_construct_ready(self):
        if self.is_filter_construct_ready() and \
                self.__padding is not None:
            return True
        return False

    def init_device_name(self, device, force=True):
        if self.__device is None or force is True:
            if device is not None:
                self.__device = device

    def get_device_name(self):
        return self.__device

    def set_filter_arch(self, filter_arch):
        self.__filters_arch = filter_arch

    def set_bias_arch(self, bias_arch):
        self.__bias_arch = bias_arch

    def set_strides_arch(self, strides_arch):
        self.__strides_arch = strides_arch

    def set_padding(self, padding):
        self.__padding = padding

    def construct_filters(self, create_network_flag, device=None, filter_list=None, filter_subset=None):
        """

        :param create_network_flag: says if the network is being created or simply retrieved.
        :param device:
        :param filter_list:
        :return:
        """

        # print ("hit!")
        self.init_device_name(device)

        if not self.is_filter_construct_ready():
            # print ("hit")
            return False

        with tf.device(self.get_device_name()):
            if filter_subset is None:
                filter_subset = range(0, len(self.__filters_arch))
            if create_network_flag is True:
                for i in filter_subset:
                    # print (self.__filters_arch[i])
                    # print ("Hit 1")

                    self.__filters.append(tf.Variable(initial_value=tf.truncated_normal(shape=self.__filters_arch[i]),
                                                      name=self.__name + "_W_" + str(i)))
                    self.__biases.append(tf.Variable(initial_value=tf.truncated_normal(shape=self.__bias_arch[i]),
                                                     name=self.__name + "_B_" + str(i)))
            elif filter_list is None:
                for i in filter_subset:
                    # print ("hit 2")
                    # print (self.__filters_arch[i])
                    self.__filters.append(tf.get_variable(name=self.__name + "_W_" + str(i),
                                                          shape=self.__filters_arch[i]))
                    self.__biases.append(tf.get_variable(name=self.__name + "_B_" + str(i),
                                                         shape=self.__bias_arch[i]))
            else:
                # print ("hit 3")
                self.__filters = filter_list

        return True

    def construct_layers(self, device=None, add_input_layer=True, filter_subset=None):
        """
        Constructs convolution layers. Can be called multiple times, to extend the size
        of the network. But in case it is done, `add_input_layer` must be set to `False`
        for all the following calls. `filter_subset` can be used to directly assign a
        set of already constructed filters. This is allowed to help the system construct
        multiple architectures with the same set of filters.
        :param device:
        :param add_input_layer:
        :param force_valid_output_range:
        :param filter_subset:
        :return:
        """
        self.init_device_name(device)

        if not self.is_layer_construct_ready():
            return False

        with tf.device(self.get_device_name()):
            if add_input_layer:
                cur_layer = tf.nn.conv2d(input=self.__network_input, filter=self.__filters[0],
                                         strides=self.__strides_arch[0], padding=self.__padding)
                cur_layer = tf.add(cur_layer, self.__biases[0])
                cur_layer = tf.nn.relu(cur_layer)
                self.__conv_layers.append(cur_layer)
            else:
                cur_layer = self.__conv_layers[len(self.__conv_layers) - 1]

            if filter_subset is None:
                filter_subset = range(1, len(self.__filters))

            for i in filter_subset:
                cur_layer = tf.nn.conv2d(input=cur_layer, filter=self.__filters[i],
                                         strides=self.__strides_arch[i], padding=self.__padding)
                cur_layer = tf.add(cur_layer, self.__biases[i])
                cur_layer = tf.nn.relu(cur_layer)
                self.__conv_layers.append(cur_layer)

            # if force_valid_output_range:
            #     output_element = self.__conv_layers[len(self.__conv_layers) -1]
            #     output_element = tf.divide(output_element, tf.reduce_max(output_element))
            #     self.__conv_layers.pop()
            #     self.__conv_layers.append(output_element)

        return True

    def check_if_file_exists(self):
        return os.path.isfile(self.__param_path)

    def restore_prams(self, session):
        if self.check_if_file_exists() and self.__saver is not None:
            self.__saver.restore(session, self.__param_path)
            return True
        return False

    def save_prams(self, session):
        if self.__saver is not None:
            self.__saver.save(session, self.__param_path)

    def get_or_init_network_output(self, output_index=None):
        """
        Constructs or obtains the output given a layer. This ensures that the
        output layer is fully constructed before returing. Upon failure it throws
        the `InvalidOutputLayer` exception.
        :param output_index: This accepts either a list or a single integer value.
                             In case a list is give, the operation is performed for
                             all the values in the list and thus, returns another
                             list with its corresponding results. The value represents
                             the layer's index. For a negative number, the index
                             is resolved to be self.number_of_layers() - output_index -1
        :return:
        """
        if isinstance(output_index, list):
            output_index_list = []

            for out_index in output_index:
                output_index_list.append(self.__get_or_init_network_output(output_index=out_index))

            return output_index_list
        else:
            return self.__get_or_init_network_output(output_index=output_index)

    def __get_or_init_network_output(self, output_index=None):

        if output_index is None:
            output_index = self.number_of_layers() -1

        if abs(output_index) <= self.number_of_layers():
            if output_index not in self.__output_layers:
                output_element = self.__conv_layers[output_index]
                output_element = tf.divide(output_element, tf.reduce_max(output_element))
                self.__output_layers[output_index] = output_element

            return self.__output_layers[output_index]

        raise InvalidOutputLayer("Error, the index for obtaining the output layer is invalid")

    def get_last_layer(self):
        # return self.__conv_layers[len(self.__conv_layers) - 1]
        return self.get_or_init_network_output(-1)

    def set_rms_network_loss(self, reset=False):
        if SRNetworkManager.RMS_ERROR not in self.__loss_network or reset:
            reduce_rms_loss = tf.reduce_mean(
                tf.square(
                    tf.subtract(self.get_last_layer(), self.__network_expected_out)))
            self.__loss_network[SRNetworkManager.RMS_ERROR] = reduce_rms_loss
            return True
        return False

    def get_rms_network_loss(self, reset=False, deep_reset=False):
        if SRNetworkManager.RMS_ERROR not in self.__loss_network or reset or deep_reset:
            self.set_rms_network_loss(deep_reset)
        return self.__loss_network[SRNetworkManager.RMS_ERROR]

    def set_psnr_network_loss(self, reset=False, deep_reset=False):

        if SRNetworkManager.PSNR_ERROR not in self.__loss_network or reset or deep_reset:
            rms_error_loss = self.get_rms_network_loss(deep_reset, deep_reset)
            reduce_psnr_loss = log10(tf.divide(1, rms_error_loss))
            reduce_psnr_loss = tf.multiply(-20.0, reduce_psnr_loss)
            self.__loss_network[SRNetworkManager.PSNR_ERROR] = reduce_psnr_loss
            return True
        return False

    def get_psnr_network_loss(self, reset=False, deep_reset=False):
        if SRNetworkManager.PSNR_ERROR not in self.__loss_network or reset or deep_reset:
            self.set_psnr_network_loss(deep_reset, deep_reset)
        return self.__loss_network[SRNetworkManager.PSNR_ERROR]

    def set_input_learner_margin(self, error_loss_flag=PSNR_ERROR, reset=False, deep_reset=False):
        if error_loss_flag == SRNetworkManager.PSNR_ERROR:
            error_loss = self.get_psnr_network_loss(deep_reset, deep_reset)
        elif error_loss_flag == SRNetworkManager.RMS_ERROR:
            error_loss = self.get_rms_network_loss(deep_reset, deep_reset)
        else:
            return False

        current_flag = SRNetworkManager.TRANSFORM_RESTRAIN_ERROR + "_" + error_loss_flag

        if current_flag not in self.__loss_network or reset or deep_reset:
            prevent_learning_input = tf.abs(tf.subtract(self.__network_expected_out, self.__network_input))
            prevent_learning_input = tf.subtract(prevent_learning_input,
                                                 tf.abs(tf.subtract(self.get_last_layer(), self.__network_input)))
            prevent_learning_input = tf.reduce_mean(tf.square(prevent_learning_input))

            error_loss = tf.add(error_loss, prevent_learning_input)

            self.__loss_network[current_flag] = error_loss
            return True

        return False

    def get_input_transform_restrain_loss(self, error_loss_flag=PSNR_ERROR, reset=False, deep_reset=False):
        current_flag = SRNetworkManager.TRANSFORM_RESTRAIN_ERROR + "_" + error_loss_flag

        if current_flag not in self.__loss_network or reset or deep_reset:
            if self.set_input_learner_margin(error_loss_flag, deep_reset, deep_reset):
                cur_loss = self.__loss_network[current_flag]
                return cur_loss
            else:
                return None

        return self.__loss_network[current_flag]

    def get_input(self):
        return self.__network_input

    def get_expected_out(self):
        return self.__network_expected_out

    def get_filters(self):
        return self.__filters

    @staticmethod
    def generate_strides_one(size):
        result = []

        for i in range(size):
            result.append([1, 1, 1, 1])

        return result

    def check_if_network_is_configured(self):
        if len(self.__conv_layers) > 0:
            return True
        return False

    def set_adam_loss_optimizer(self, network_loss, var_list=None):
        if var_list is None:
            var_list = []
            var_list.extend(self.__filters)
            var_list.extend(self.__biases)
        if self.check_if_network_is_configured():
            self.__optimizer = tf.train.AdamOptimizer().minimize(network_loss, var_list=var_list)

    def get_adam_loss_optimizer(self, network_loss, var_list=None):
        if self.__optimizer is None:
            self.set_adam_loss_optimizer(network_loss, var_list)
        return self.__optimizer


def main():
    img_manager = ImageDSManage(["/home/sreramk/PycharmProjects/neuralwithtensorgpu/dataset/DIV2K_train_HR/"],
                                image_buffer_limit=100, buffer_priority=100)

    weights_file_name = "w"

    network_manager = SRNetworkManager("model_1", "/home/sreramk/PycharmProjects/neuralwithtensorgpu/dataset/DIV2K_train_HR/")
    network_manager.set_strides_arch(SRNetworkManager.generate_strides_one(3))

    filter_arch = [
        [10, 10, 3, 80],
        [2, 2, 80, 40],
        [10, 10, 40, 3]
    ]

    network_manager.set_filter_arch(filter_arch)

    bias_arch = [
        [80], [40], [3]
    ]

    network_manager.set_bias_arch(bias_arch)

    network_manager.get_device_name()

    network_manager.init_device_name('/gpu:0')

    network_manager.construct_filters(create_network_flag=network_manager.check_if_file_exists())

    network_manager.construct_layers()

    network_loss = network_manager.get_input_transform_restrain_loss()

    adam_optimizer = network_manager.get_adam_loss_optimizer(network_loss)

    network_manager.get_or_init_network_output([None, -2])

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
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
            tstart = None
            for i in range(10000):
                # for j in range(10):

                min_x, min_y, batch_down_sampled, batch_original = img_manager.get_batch(batch_size=6,
                                                                                         down_sample_factor=4,
                                                                                         min_x_f=70,
                                                                                         min_y_f=70)

                minimize, loss = sess.run(fetches=[adam_optimizer, network_loss],
                                          feed_dict={network_manager.get_input(): batch_down_sampled,
                                                     network_manager.get_expected_out(): batch_original,
                                                     })
                if i % 100 == 0:
                    print("epoch :" + str(i))
                    print("loss: " + str(loss))
                    tend = datetime.now()
                    if tstart is None:
                        tstart = tend
                    print ("Time = " + str(tend - tstart))
                    tstart = datetime.now()

            for i in range(1):
                # min_x, min_y, batch_down_sampled, batch_original = img_manager.get_batch(batch_size=1, down_sample_factor=10,
                #                                                                     min_x_f=400, min_y_f=400 )

                computed_image = sess.run(fetches=[network_manager.get_or_init_network_output()],
                                          feed_dict={network_manager.get_input(): batch_down_sampled})

                print(len(computed_image))
                print(len(computed_image[0]))
                print(len(computed_image[0][0]))
                print(len(computed_image[0][0][0]))
                print(len(computed_image[0][0][0][0]))

                #cv2.imshow("original" + str(i) + "_" + str(j), batch_original[i])
                #cv2.imshow("down_sampled" + str(i) + "_" + str(j), batch_down_sampled[i])
                #cv2.imshow("computed" + str(i) + "_" + str(j), computed_image[0][i])

            #cv2.waitKey(0)

            computed_image = None

            for i in range(2):
                min_x, min_y, batch_down_sampled, batch_original = img_manager.get_batch(batch_size=1,
                                                                                         down_sample_factor=4,
                                                                                         min_x_f=400, min_y_f=400)

                # batch_down_sampled = 1 - np.asarray(batch_down_sampled)

                # if computed_image != None:
                #    batch_down_sampled[0] = computed_image[0][0]

                computed_image = sess.run(fetches=[network_manager.get_or_init_network_output()],
                                          feed_dict={network_manager.get_input(): batch_down_sampled})

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

            for i in range(2):
                min_x, min_y, batch_down_sampled, batch_original = img_manager.get_batch(batch_size=1,
                                                                                         down_sample_factor=4,
                                                                                         min_x_f=400, min_y_f=400)

                # batch_down_sampled = 1 - np.asarray(batch_down_sampled)

                # if computed_image != None:
                #    batch_down_sampled[0] = computed_image[0][0]

                computed_image, computed_org = sess.run(fetches=[
                                                   network_manager.get_or_init_network_output(output_index=-2),
                                                   network_manager.get_or_init_network_output()],
                                          feed_dict={network_manager.get_input(): batch_down_sampled})

                """
                print (len(computed_image))
                print (len(computed_image[0]))
                print (len(computed_image[0][0]))
                print (len(computed_image[0][0][0]))
                print (len(computed_image[0][0][0][0]))
                """

                cv2.imshow("L2_2original" + str(i) + "_" + str(j), batch_original[0])
                cv2.imshow("L2_2down_sampled" + str(i) + "_" + str(j), batch_down_sampled[0])
                cv2.imshow("L2_2computed" + str(i) + "_" + str(j), computed_org[0])
                for x in range(0, int(40)):
                    computed_image_temp = computed_image[0][:, :, x]
                    cv2.imshow("L2_2computed" + str(i) + "_" + str(j) +"_" + str(x), computed_image_temp)

            cv2.waitKey(0)
            cv2.destroyAllWindows()


main()
