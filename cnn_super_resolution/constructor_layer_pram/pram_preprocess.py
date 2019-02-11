# copyright (c) 2019 K Sreram, all rights reserved

import tensorflow as tf

def preprocess_cnn_parameters(self, device):
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

    with tf.device(device):
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