# Copyright (c) 2019 K Sreram, All rights reserved

import os

import tensorflow as tf

from prepare_dataset.img_ds_manage import ImageDSManage


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


class SRNetworkManager:
    RMS_ERROR = "RMS_ERROR"
    PSNR_ERROR = "PSNR_ERROR"
    INPUT_LEARNER_MARGIN = "INPUT_LEARNER_MARGIN"

    def __init__(self, filters_arch, bias_arch, strides_arch,
                 device, name, param_path, padding="SAME"):
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

        with tf.device(self.__device):
            self.__network_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None],
                                                  name=self.__name + "_input")
            self.__network_expected_out = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None],
                                                         name=self.__name + "_expected_out")

        self.__saver = tf.train.Saver()

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

    def construct_filters(self, create_network_flag, device=None, filter_list=None):

        self.init_device_name(device)

        with tf.device(self.get_device_name()):
            if create_network_flag is True:
                for i in range(len(self.__filters_arch)):
                    self.__filters.append(tf.Variable(initial_value=tf.truncated_normal(shape=self.__filters_arch[i]),
                                                      name=self.__name + "_W_" + str(i)))
                    self.__biases.append(tf.Variable(initial_value=tf.truncated_normal(shape=self.__bias_arch[i]),
                                                     name=self.__name + "_B_" + str(i)))
            elif filter_list is None:
                for i in range(len(self.__filters_arch)):
                    self.__filters.append(tf.get_variable(name=self.__name + "_W_" + str(i),
                                                          shape=self.__filters_arch[i]))
                    self.__biases.append(tf.get_variable(name=self.__name + "_B_" + str(i),
                                                         shape=self.__bias_arch[i]))
            else:
                self.__filters = filter_list

    def construct_layers(self, device=None, add_input_layer=True, filter_subset=None):

        self.init_device_name(device)

        with tf.device(self.get_device_name()):
            if add_input_layer:
                cur_layer = tf.nn.conv2d(input=self.__network_input, filter=self.__filters[0],
                                         strides=self.__strides_arch[0], padding=self.__padding)
                cur_layer = tf.add(cur_layer, self.__biases[0])
                self.__conv_layers.append(cur_layer)
            else:
                cur_layer = self.__conv_layers[len(self.__conv_layers) - 1]

            if filter_subset is None:
                filter_subset = range(1, len(self.__filters))

            for i in filter_subset:
                cur_layer = tf.nn.conv2d(input=cur_layer, filter=self.__filters[i],
                                         strides=self.__strides_arch[i], padding=self.__padding)
                cur_layer = tf.add(cur_layer, self.__biases[i])
                self.__conv_layers.append(cur_layer)

    def construct_network(self, create_network_flag, device, filter_list=None,
                          add_input_layer=True, filter_subset=None):

        self.construct_filters(create_network_flag, device, filter_list)
        self.construct_layers(device, add_input_layer, filter_subset)

    def restore_prams(self, session):
        if os.path.isfile(self.__param_path):
            self.__saver.restore(session, self.__param_path)
            return True
        return False

    def save_prams(self, session):
        self.__saver.save(session, self.__param_path)

    def get_network_output(self):
        return self.__conv_layers[len(self.__conv_layers) - 1]

    def get_last_layer(self):
        return self.__conv_layers[len(self.__conv_layers) - 1]

    def set_rms_network_loss(self):
        if SRNetworkManager.RMS_ERROR not in self.__loss_network:
            reduce_rms_loss = tf.reduce_mean(
                tf.square(
                    tf.subtract(self.get_last_layer(), self.__network_expected_out)))
            self.__loss_network[SRNetworkManager.RMS_ERROR] = reduce_rms_loss
            return True
        return False

    def get_rms_network_loss(self):
        if SRNetworkManager.RMS_ERROR not in self.__loss_network:
            self.set_rms_network_loss()
        return self.__loss_network[SRNetworkManager.RMS_ERROR]

    def set_psnr_network_loss(self):

        if SRNetworkManager.PSNR_ERROR not in self.__loss_network:
            rms_error_loss = self.get_rms_network_loss()
            reduce_psnr_loss = log10(tf.divide(1, rms_error_loss))
            reduce_psnr_loss = tf.multiply(-20.0, reduce_psnr_loss)
            self.__network_input[SRNetworkManager.PSNR_ERROR] = reduce_psnr_loss
            return True
        return False

    def get_psnr_network_loss(self):
        if SRNetworkManager.PSNR_ERROR not in self.__loss_network:
            self.set_psnr_network_loss()
        return self.__loss_network[SRNetworkManager.PSNR_ERROR]

    def set_input_learner_margin(self, error_loss_flag=PSNR_ERROR):
        if error_loss_flag == SRNetworkManager.PSNR_ERROR:
            error_loss = self.get_psnr_network_loss()
        elif error_loss_flag == SRNetworkManager.RMS_ERROR:
            error_loss = self.get_rms_network_loss()
        else:
            return False

        CURRENT_FLAG = SRNetworkManager.INPUT_LEARNER_MARGIN + "_" + error_loss_flag

        if CURRENT_FLAG not in self.__loss_network:
            prevent_learning_input = tf.abs(tf.subtract(self.__network_expected_out, self.__network_input))
            prevent_learning_input = tf.subtract(prevent_learning_input,
                                                 tf.abs(tf.subtract(self.get_last_layer(), self.__network_input)))
            prevent_learning_input = tf.reduce_mean(tf.square(prevent_learning_input))

            error_loss = tf.add(error_loss, prevent_learning_input)

            self.__loss_network[CURRENT_FLAG] = error_loss
            return True

        return False

    def get_input_learner_margin(self, error_loss_flag=PSNR_ERROR):
        CURRENT_FLAG = SRNetworkManager.INPUT_LEARNER_MARGIN + "_" + error_loss_flag

        if CURRENT_FLAG not in self.__loss_network:
            if self.set_input_learner_margin(error_loss_flag):
                cur_loss = self.__loss_network[CURRENT_FLAG]
                return cur_loss
            else:
                return None

        return self.__loss_network[CURRENT_FLAG]

    def get_input(self):
        return self.__network_input

    def get_expected_out(self):
        return self.__network_expected_out

    def get_filters(self):
        return self.__filters


if __name__ == "__main__":
    img_manager = ImageDSManage(["/home/sreramk/PycharmProjects/neuralwithtensorgpu/dataset/DIV2K_train_HR/"],
                                image_buffer_limit=100)

    network_manager = SRNetworkManager

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        pass
