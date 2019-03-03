# copyright (c) 2019 K Sreram, All rights reserved.
import cv2
import os

import numpy

from prepare_dataset.sr_image_ds_manager import ImageDSManage
from trainer.exceptions import LossUninitialized, LossOptimizerUninitialized, RequiredPlaceHolderNotInitialized, \
    TrainDatasetNotInitialized, TestDatasetNotInitialized, SaveInstanceNotInitialized, RunTestError, \
    ModelNotInitialized, ParameterNotInitialized, InvalidArgumentCombination
from trainer.model_base import ModelBase
import tensorflow as tf
from model_saver_manager import model_saver


class SRModelPSNR(ModelBase):
    RMS_TRIPLET_LOSS = "rms_triplet_loss"
    RMS_LOSS = "rms_loss"

    def __init__(self):
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

        self.__train_ds_manage = None
        self.__test_ds_manage = None

        self.__adam_rms = None

        self.__adam_rms_triplet = None

        self.__model_saver_inst = None

        super().__init__()

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
        self.__adam_rms_triplet = tf.train.AdamOptimizer().minimize(self.__rms_triplet_loss, var_list=self.__parameters)
        return self.__rms_triplet_loss

    def __create_rms_loss(self):
        loss = tf.square(tf.subtract(self.__expected_output_data, self.__model))
        loss = tf.reduce_mean(loss)
        self.__rms_loss = loss
        self.__adam_rms = tf.train.AdamOptimizer().minimize(self.__rms_loss, var_list=self.__parameters)
        return self.__rms_loss

    def __get_input_data_placeholder(self):
        if self.__input_data is None:
            raise RequiredPlaceHolderNotInitialized
        return self.__input_data

    def __get_expected_output_placeholder(self):
        if self.__expected_output_data is None:
            raise RequiredPlaceHolderNotInitialized
        return self.__expected_output_data

    def __get_rms_loss(self):
        if self.__rms_loss is None:
            raise LossUninitialized
        return self.__rms_loss

    def __get_rms_triplet_loss(self):
        if self.__rms_triplet_loss is None:
            raise LossUninitialized
        return self.__rms_triplet_loss

    def __get_rms_loss_adam_opt(self):
        if self.__adam_rms is None:
            raise LossOptimizerUninitialized
        return self.__adam_rms

    def __get_rms_triplet_loss_adam_opt(self):
        if self.__adam_rms_triplet is None:
            raise LossOptimizerUninitialized
        return self.__adam_rms_triplet

    def set_rms_triplet_loss(self):
        self.__active_loss = SRModelPSNR.RMS_TRIPLET_LOSS

    def set_rms_loss(self):
        self.__active_loss = SRModelPSNR.RMS_LOSS

    def is_rms_triplet_loss_active(self):
        return self.__active_loss == SRModelPSNR.RMS_TRIPLET_LOSS

    def is_rms_loss_active(self):
        return self.__active_loss == SRModelPSNR.RMS_LOSS

    def __get_active_loss(self):
        return self.__active_loss

    def set_model_saver_inst(self, model_saver_inst):
        self.__model_saver_inst = model_saver_inst

    def prepare_train_test_dataset(self, train_dataset_path, test_dataset_path,
                                   image_buffer_limit_train=int(50 * 0.8), image_buffer_limit_test=int(50 * 0.2)):
        self.__train_ds_manage = ImageDSManage(train_dataset_path,
                                               image_buffer_limit=image_buffer_limit_train, buffer_priority=0.01,
                                               buffer_priority_acceleration=0.01,
                                               buffer_priority_cap=1000)

        self.__test_ds_manage = ImageDSManage(test_dataset_path,
                                              image_buffer_limit=image_buffer_limit_test, buffer_priority=0.01,
                                              buffer_priority_acceleration=0.01,
                                              buffer_priority_cap=1000)

    def run_train(self,
                  # parameters for train:
                  num_of_epochs=10, single_epoch_count=12010, checkpoint_iteration_count=3000,
                  display_status=True, display_status_iter=100, batch_size=6, down_sample_factor=4,
                  min_x_f=100, min_y_f=100,

                  # parameters for test:
                  terminal_loss=0.00001, test_min_x_f=500, test_min_y_f=500, number_of_samples=int(50 * 0.2),
                  run_test_periodically=3000, execute_tests=True,

                  # parameters for checkpoint:
                  save_checkpoints=True):
        """

        :param test_min_y_f:
        :param test_min_x_f:
        :param number_of_samples:
        :param execute_tests:
        :param run_test_periodically: Number of iterations to wait before running the test
        :param save_checkpoints:
        :param num_of_epochs:
        :param single_epoch_count:
        :param checkpoint_iteration_count:
        :param display_status:
        :param display_status_iter:
        :param batch_size:
        :param down_sample_factor:
        :param min_x_f:
        :param min_y_f:
        :param terminal_loss:
        :return:
        """
        ModelSaver = model_saver.ModelSaver
        if not isinstance(self.__train_ds_manage, ImageDSManage):
            raise TrainDatasetNotInitialized

        if save_checkpoints and not isinstance(self.__model_saver_inst, model_saver.ModelSaver):
            raise SaveInstanceNotInitialized

        if self.is_rms_loss_active():
            optimizer = self.__get_rms_loss_adam_opt()
        else:
            optimizer = self.__get_rms_triplet_loss_adam_opt()

        if self.is_rms_loss_active():
            train_loss = self.__get_rms_loss()
        else:
            train_loss = self.__get_rms_triplet_loss()

        def print_checkpoint():
            print("Checkpoint committed")

        def execute_all_epoch():
            with tf.Session() as sess:
                init = tf.initialize_all_variables()
                sess.run(init)
                iteration_count = 0
                total_iterations = num_of_epochs * single_epoch_count

                def get_train_loss():
                    return sess.run(fetches=[train_loss],
                                    feed_dict={self.__get_input_data_placeholder(): batch_down_sampled,
                                               self.__get_expected_output_placeholder(): batch_original})

                for epoch in range(num_of_epochs):

                    for i in range(single_epoch_count):

                        result = self.__train_ds_manage.get_batch(batch_size=batch_size,
                                                                  down_sample_factor=down_sample_factor,
                                                                  min_x_f=min_x_f,
                                                                  min_y_f=min_y_f)

                        min_x, min_y, batch_down_sampled, batch_original = result

                        self.__train_ds_manage.acceleration_step()

                        sess.run(fetches=[optimizer],
                                 feed_dict={self.__get_input_data_placeholder(): batch_down_sampled,
                                            self.__get_expected_output_placeholder(): batch_original})

                        if display_status and iteration_count % display_status_iter == 0:
                            cur_train_loss = get_train_loss()
                            print("Epoch: %d, epoch train percentage : %.3f, "
                                  "total train percentage : %.3f training loss: %f" %
                                  (epoch, ((float(i) / single_epoch_count) * 100),
                                   ((float(iteration_count) / total_iterations) * 100), cur_train_loss))

                        if save_checkpoints:
                            cur_train_loss = get_train_loss()
                            self.__model_saver_inst.checkpoint_model(float(cur_train_loss), sess=sess,
                                                                     skip_type=ModelSaver.TimeIterSkipManager.ST_ITER_SKIP,
                                                                     skip_duration=checkpoint_iteration_count,
                                                                     exec_on_checkpoint=print_checkpoint)
                        if execute_tests and iteration_count % run_test_periodically == 0:
                            cur_test_loss = \
                                self.run_test(test_min_x_f, test_min_y_f, number_of_samples, sess, down_sample_factor)

                            if cur_test_loss < terminal_loss:
                                return

                        iteration_count += 1

                    # display at the end of epoch, if not displayed
                    if display_status and iteration_count % display_status_iter != 0:
                        cur_train_loss = get_train_loss()
                        print("Epoch: %d, total train percentage : %.3f, training loss: %f" %
                              (epoch, ((float(iteration_count) / total_iterations) * 100), cur_train_loss))

        execute_all_epoch()
        if execute_tests:
            print("Final loss:")
            with tf.Session() as sess:
                    self.run_test(test_min_x_f, test_min_y_f, number_of_samples, sess, down_sample_factor)


    def run_test(self, min_x_f=500, min_y_f=500, number_of_samples=int(50 * 0.2), sess=None, down_sample_factor=4):
        """

        :param down_sample_factor:
        :param sess:
        :param min_x_f:
        :param min_y_f:
        :param number_of_samples:
        :return:
        """
        if not isinstance(self.__test_ds_manage, ImageDSManage):
            raise TestDatasetNotInitialized

        if self.is_rms_loss_active():
            test_loss = self.__get_rms_loss()
        else:
            test_loss = self.__get_rms_triplet_loss()

        result = self.__test_ds_manage.get_batch(batch_size=number_of_samples,
                                                 down_sample_factor=down_sample_factor,
                                                 min_x_f=min_x_f,
                                                 min_y_f=min_y_f)

        min_x, min_y, batch_down_sampled, batch_original = result

        session_is_parent = True if sess is None else False

        if not session_is_parent:
            sess = tf.Session()

        def get_test_loss():
            return sess.run(fetches=[test_loss],
                            feed_dict={self.__get_input_data_placeholder(): batch_down_sampled,
                                       self.__get_expected_output_placeholder(): batch_original})

        cur_test_loss = get_test_loss()

        print("Test loss: %.3f, Total samples checked with : %d" % (cur_test_loss, number_of_samples))

        if not session_is_parent:
            sess.close()

        return  cur_test_loss

    def execute_model(self, model_input=None, model_input_path=None, sess=None):

        session_is_parent = True if sess is None else False

        if not session_is_parent:
            sess = tf.Session()

        if (model_input is None and model_input_path is None) or \
                (model_input is not None and model_input_path is not None):
            raise InvalidArgumentCombination

        if model_input_path is not None:
            model_input = cv2.imread(model_input_path)
            model_input = numpy.float32(model_input) / 255.0
            model_input = [model_input]
            model_input = ImageDSManage.ensure_numpy_array(model_input)

        img_result = sess.run(fetches=[self.get_model()],
                              feed_dict={self.__get_input_data_placeholder(): model_input})

        if not session_is_parent:
            sess.close()

        return img_result

    def get_model(self, **args):
        if self.__model is None:
            raise ModelNotInitialized
        return self.__model

    def get_parameter_tensors(self):
        if self.__parameters is None:
            raise ParameterNotInitialized

        return self.__parameters
