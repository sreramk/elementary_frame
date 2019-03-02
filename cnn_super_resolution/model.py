# copyright (c) 2019 K Sreram, All rights reserved.
import os

from trainer.model_base import ModelBase
import tensorflow as tf
from model_saver_manager import model_saver

ModelSaver = model_saver.ModelSaver


class SRModelPSNR(ModelBase):
    RMS_TRIPLET_LOSS = "rms_triplet_loss"
    RMS_LOSS = "rms_loss"

    def __init__(self, save_name, save_file_path=os.getcwd(),
                 check_point_digits=5, extension=ModelSaver.DEFAULT_EXTENSION, reset=False,
                 model_type=ModelSaver.MT_TENSORFLOW):
        self.__input_data = None
        self.__expected_output_data = None

        self.__filters = []

        self.__biases = []

        self.__model = None

        self.__triplet_loss = None

        self.__rms_triplet_loss = None

        self.__rms_loss = None

        self.__active_loss = SRModelPSNR.RMS_TRIPLET_LOSS

        self.__padding = "SAME"

        self.__strides = [1, 1, 1, 1]

        self.__create_place_holders()

        self.__create_parameters()

        self.__create_model()

        self.__create_triplet_loss()

        self.__create_rms_loss()

        self.__create_rms_triplet_loss()

        self.__parameters = list(self.__filters)
        self.__parameters.extend(self.__biases)

        model_saver_inst = ModelSaver(save_name, self.__parameters, save_file_path=save_file_path,
                                      check_point_digits=check_point_digits, extension=extension, reset=reset,
                                      model_type=model_type)

        super().__init__(model_saver_inst)

    def __create_place_holders(self):
        self.__input_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None])
        self.__expected_output_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None])
        return self.__input_data, self.__expected_output_data

    def __create_parameters(self):

        self.__filters.append(tf.Variable(initial_value=tf.truncated_normal(shape=[10, 10, 3, 80])))
        self.__filters.append(tf.Variable(initial_value=tf.truncated_normal(shape=[2, 2, 80, 40])))
        self.__filters.append(tf.Variable(initial_value=tf.truncated_normal(shape=[10, 10, 40, 3])))

        self.__biases.append(tf.Variable(initial_value=tf.truncated_normal(shape=[80])))
        self.__biases.append(tf.Variable(initial_value=tf.truncated_normal(shape=[40])))
        self.__biases.append(tf.Variable(initial_value=tf.truncated_normal(shape=[3])))

        return self.__filters, self.__biases

    def __create_model(self):

        if len(self.__filters) > 0 and len(self.__biases) > 0:
            cur_layer = tf.nn.conv2d(input=self.__input_data, filter=self.__filters[0],
                                     strides=self.__strides, padding=self.__padding)
            cur_layer = tf.add(cur_layer, self.__biases[0])
            cur_layer = tf.nn.relu(cur_layer)

            cur_layer = tf.nn.conv2d(input=cur_layer, filter=self.__filters[1],
                                     strides=self.__strides, padding=self.__padding)
            cur_layer = tf.add(cur_layer, self.__biases[1])
            cur_layer = tf.nn.relu(cur_layer)

            cur_layer = tf.nn.conv2d(input=cur_layer, filter=self.__filters[2],
                                     strides=self.__strides, padding=self.__padding)
            cur_layer = tf.add(cur_layer, self.__biases[2])
            cur_layer = tf.nn.relu(cur_layer)

            cur_layer = tf.divide(cur_layer, tf.reduce_max(cur_layer))

            self.__model = cur_layer

            return cur_layer

    def __create_triplet_loss(self):
        temp1 = tf.subtract(self.__input_data, self.__expected_output_data)
        temp2 = tf.subtract(self.__input_data, self.__model)
        temp1 = tf.square(temp1)
        temp2 = tf.square(temp2)

        loss = tf.subtract(temp1, temp2)
        loss = tf.square(loss)

        self.__triplet_loss = tf.reduce_mean(loss)
        return self.__triplet_loss

    def __create_rms_triplet_loss(self):
        # forces the model to not learn the identity.
        if self.__triplet_loss is None:
            self.__create_triplet_loss()

        loss = tf.square(tf.subtract(self.__expected_output_data, self.__model))
        loss = tf.reduce_mean(loss)
        self.__rms_triplet_loss = tf.add(self.__triplet_loss, loss)
        return self.__rms_triplet_loss

    def __create_rms_loss(self):
        loss = tf.square(tf.subtract(self.__expected_output_data, self.__model))
        loss = tf.reduce_mean(loss)
        self.__rms_loss = loss
        return self.__rms_loss

    def set_rms_triplet_loss(self):
        self.__active_loss = SRModelPSNR.RMS_TRIPLET_LOSS

    def set_rms_loss(self):
        self.__active_loss = SRModelPSNR.RMS_LOSS

    def prepare_training_dataset(self):
        pass

    def run_train(self, num_of_epochs, **args):
        pass

    def run_test(self, training_set_frac=1, **args):
        pass

    def execute_model(self, model_input=None, **args):
        pass

    def get_model(self, **args):
        pass

    def get_parameter_tensors(self):
        pass
