# copyright (c) 2019 K Sreram, All rights reserved.
import os
import urllib.parse

import cv2
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from os.path import isfile, join
from skimage import io

from cnn_super_resolution.sr_dataset_manage import SRDSManage
from model_saver_manager import model_saver
from model_saver_manager.exceptions import CheckpointCannotBeFirstRun
from prepare_dataset.batch_dataset_iterator import BatchDataSetIterator
from prepare_dataset.data_set_buffer import DataBuffer
from prepare_dataset.dataset_manager import DataSetManager
from trainer.exceptions import LossUninitialized, LossOptimizerUninitialized, RequiredPlaceHolderNotInitialized, \
    TrainDatasetNotInitialized, TestDatasetNotInitialized, SaveInstanceNotInitialized, ModelNotInitialized, \
    ParameterNotInitialized, InvalidArgumentCombination, InvalidType
from trainer.model_base import ModelBase
from utils.general_utils import ensure_numpy_array, check_if_numpy_array
from utils.running_avg import RunningAvg


class SRModel(ModelBase):
    RMS_TRIPLET_LOSS = "rms_triplet_loss"
    RMS_LOSS = "rms_loss"
    PSNR_LOSS = "psnr_loss"
    PSNR_TRIPLET_LOSS = "psnr_triplet_loss"

    def __init__(self):

        ## hyper parameters tuning support:
        self.__learning_rate = None
        self.__learning_rate_input = None

        self.__momentum = None
        self.__momentum_input = None

        self.__assign_learning_rate = None
        self.__assign_momentum = None

        self.__add_hyper_parameter_tuning_support()

        self.__input_data = None
        self.__expected_output_data = None

        self.__test_ds_container = None
        self.__train_ds_container = None

        self.__ds_manage = None

        self.__adam_rms = None

        self.__adam_rms_triplet = None

        self.__adam_psnr_loss = None

        self.__adam_psnr_triplet_loss = None

        self.__model_saver_inst = None

        self.__filters = []

        self.__biases = []

        self.__model = None

        self.__triplet_loss = None

        self.__rms_triplet_loss = None

        self.__rms_loss = None

        self.__psnr_loss = None

        self.__psnr_triplet_loss = None

        self.__active_loss = SRModel.RMS_TRIPLET_LOSS

        self.__padding = "SAME"

        self.__strides = [1, 1, 1, 1]

        self.__create_place_holders()

        self.__create_parameters()

        self.__create_model()

        self.__create_triplet_loss()

        self.__create_rms_loss()

        self.__create_rms_triplet_loss()

        self.__create_psnr_loss()

        self.__create_psnr_triplet_loss()

        self.__num_of_optimization_levels = 5

        super().__init__()

        self.__session = None
        self.reset_session()

    @staticmethod
    def display_image(img, black_and_white=False, invert_colors=False, figure_size=None):

        def plt_figure():
            if figure_size is not None:
                plt.figure(figsize=figure_size)
            else:
                plt.figure()

        if black_and_white:
            temp = []
            for i in range(len(img)):
                temp.append([])
                for j in range(len(img[0])):
                    temp[i].append([img[i][j], img[i][j], img[i][j]])
            img = numpy.array(temp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt_figure()
            plt.imshow(img, cmap='gray')
        else:
            if not invert_colors:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt_figure()
            plt.imshow(img)
        plt.colorbar()
        plt.grid(False)
        plt.show()

    @staticmethod
    def is_url(url):
        return urllib.parse.urlparse(url).scheme != ""

    @staticmethod
    def fetch_image(input_image_path, with_batch_column=True):

        if SRModel.is_url(input_image_path):
            input_image = io.imread(input_image_path)
        else:
            input_image = cv2.imread(input_image_path)

        input_image = numpy.float32(input_image) / 255.0

        input_image = ensure_numpy_array(input_image)

        if len(input_image.shape) == 3 and with_batch_column:
            input_image = [input_image]

        input_image = ensure_numpy_array(input_image)

        return input_image

    @staticmethod
    def get_image_dimensions(img):
        if check_if_numpy_array(img):
            size_y = len(img)
            size_x = len(img[0])
            return size_x, size_y

    @staticmethod
    def save_image(img, name):
        cv2.imwrite(name, img)

    @staticmethod
    def list_all_files_in_dir(directory):
        return [f for f in os.listdir(directory) if isfile(join(directory, f))]

    @staticmethod
    def zoom_image(img, zoom_factor_x=1.0, zoom_factor_y=1.0):
        result = cv2.resize(img,
                            dsize=(int(len(img[0]) * zoom_factor_x),
                                   int(len(img) * zoom_factor_y)),
                            interpolation=cv2.INTER_NEAREST)
        return ensure_numpy_array(result)

    @staticmethod
    def tf_repeat(tensor, repeats):
        """
        Args:

        input: A Tensor. 1-D or higher.
        repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

        Returns:

        A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
        """
        with tf.variable_scope("repeat"):
            expanded_tensor = tf.expand_dims(tensor, -1)
            multiples = [1] + repeats
            tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
            repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
        return repeated_tensor

    def close_session(self):
        if self.__session is not None:
            self.__session.close()

    def reset_session(self):
        self.close_session()
        self.__session = tf.Session()
        init = tf.initialize_all_variables()
        self.__session.run(init)
        if self.__model_saver_inst is not None:
            self.__model_saver_inst.load_checkpoint(sess=self.__session)

    def __del__(self):
        self.close_session()

    def get_session(self):
        return self.__session

    def __add_hyper_parameter_tuning_support(self):
        self.__learning_rate = tf.Variable(initial_value=0.001, expected_shape=(), dtype=tf.float32)
        self.__momentum = tf.Variable(initial_value=0.001, expected_shape=(), dtype=tf.float32)
        self.__learning_rate_input = tf.placeholder(dtype=tf.float32, shape=())
        self.__momentum_input = tf.placeholder(dtype=tf.float32, shape=())
        self.__assign_learning_rate = tf.assign(self.__learning_rate, self.__learning_rate_input)
        self.__assign_momentum = tf.assign(self.__momentum, self.__momentum_input)

    def __create_place_holders(self):
        self.__input_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None])
        self.__expected_output_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None])
        return self.__input_data, self.__expected_output_data

    def __create_parameters(self):

        self.__filters.append(tf.Variable(initial_value=tf.truncated_normal(shape=[10, 10, 3, 32])))
        self.__filters.append(tf.Variable(initial_value=tf.truncated_normal(shape=[8, 8, 32, 64])))

        self.__filters.append(tf.Variable(initial_value=tf.truncated_normal(shape=[4, 4, 64, 32])))
        self.__filters.append(tf.Variable(initial_value=tf.truncated_normal(shape=[2, 2, 32, 64])))

        self.__filters.append(tf.Variable(initial_value=tf.truncated_normal(shape=[6, 6, 64, 32])))
        self.__filters.append(tf.Variable(initial_value=tf.truncated_normal(shape=[10, 10, 32, 3])))

        self.__biases.append(tf.Variable(initial_value=tf.truncated_normal(shape=[32])))
        self.__biases.append(tf.Variable(initial_value=tf.truncated_normal(shape=[64])))
        self.__biases.append(tf.Variable(initial_value=tf.truncated_normal(shape=[32])))
        self.__biases.append(tf.Variable(initial_value=tf.truncated_normal(shape=[64])))
        self.__biases.append(tf.Variable(initial_value=tf.truncated_normal(shape=[32])))
        self.__biases.append(tf.Variable(initial_value=tf.truncated_normal(shape=[3])))

        self.__parameters = list(self.__filters)
        self.__parameters.extend(self.__biases)

        return self.__filters, self.__biases, self.__parameters

    def __create_model(self):

        if len(self.__filters) > 0 and len(self.__biases) > 0:
            variance_epsilon = tf.constant(0.00000001, shape=())
            cur_layer1 = tf.nn.conv2d(input=self.__input_data, filter=self.__filters[0],
                                      strides=self.__strides, padding=self.__padding)
            cur_layer1 = tf.add(cur_layer1, self.__biases[0])
            # cur_layer1 = tf.nn.relu(cur_layer1)
            # mean, variance = tf.nn.moments(cur_layer1, axes=[0, 1, 2])
            # cur_layer1 = tf.nn.batch_normalization(cur_layer1, mean, variance, offset=None,
            #                                       scale=None, variance_epsilon=variance_epsilon)

            cur_layer2 = tf.nn.conv2d(input=cur_layer1, filter=self.__filters[1],
                                      strides=self.__strides, padding=self.__padding)
            cur_layer2 = tf.add(cur_layer2, self.__biases[1])
            # cur_layer2 = tf.nn.relu(cur_layer2)
            # mean, variance = tf.nn.moments(cur_layer2, axes=[0, 1, 2])
            # cur_layer2 = tf.nn.batch_normalization(cur_layer2, mean, variance, offset=None,
            #                                       scale=None, variance_epsilon=variance_epsilon)

            cur_layer3 = tf.nn.conv2d(input=cur_layer2, filter=self.__filters[2],
                                      strides=self.__strides, padding=self.__padding)
            cur_layer3 = tf.add(cur_layer3, self.__biases[2])
            # cur_layer3 = tf.nn.relu(cur_layer3)
            # mean, variance = tf.nn.moments(cur_layer3, axes=[0, 1, 2])
            # cur_layer3 = tf.nn.batch_normalization(cur_layer3, mean, variance, offset=None,
            #                                       scale=None, variance_epsilon=variance_epsilon)

            cur_layer3 = tf.add(cur_layer1, cur_layer3)
            # cur_layer3 = tf.divide(cur_layer3, 2.0)

            cur_layer4 = tf.nn.conv2d(input=cur_layer3, filter=self.__filters[3],
                                      strides=self.__strides, padding=self.__padding)
            cur_layer4 = tf.add(cur_layer4, self.__biases[3])
            # cur_layer4 = tf.nn.relu(cur_layer4)
            # mean, variance = tf.nn.moments(cur_layer4, axes=[0, 1, 2])
            # cur_layer4 = tf.nn.batch_normalization(cur_layer4, mean, variance, offset=None,
            #                                      scale=None, variance_epsilon=variance_epsilon)
            # cur_layer4 = tf.multiply(cur_layer1, cur_layer4)
            # cur_layer4 = tf.divide(cur_layer4, 2.0)
            # cur_layer4 = tf.nn.relu(cur_layer4)

            cur_layer5 = tf.nn.conv2d(input=cur_layer4, filter=self.__filters[4],
                                      strides=self.__strides, padding=self.__padding)
            cur_layer5 = tf.add(cur_layer5, self.__biases[4])
            # cur_layer5 = tf.nn.relu(cur_layer5)
            # mean, variance = tf.nn.moments(cur_layer5, axes=[0, 1, 2])
            # cur_layer5 = tf.nn.batch_normalization(cur_layer5, mean, variance, offset=None,
            #                                       scale=None, variance_epsilon=variance_epsilon)

            cur_layer5 = tf.add(cur_layer1, cur_layer5)

            cur_layer6 = tf.nn.conv2d(input=cur_layer5, filter=self.__filters[5],
                                      strides=self.__strides, padding=self.__padding)
            cur_layer6 = tf.add(cur_layer6, self.__biases[5])
            cur_layer6 = tf.nn.relu(cur_layer6)

            cur_layer6 = tf.divide(cur_layer6, tf.reduce_max(cur_layer6))

            self.__model = cur_layer6

            return self.__model

    def __create_triplet_loss(self):
        temp1 = tf.subtract(self.__input_data, self.__expected_output_data)
        temp2 = tf.subtract(self.__input_data, self.__model)
        temp1 = tf.square(temp1)
        temp2 = tf.square(temp2)

        loss = tf.subtract(temp1, temp2)
        loss = tf.square(loss)

        self.__triplet_loss = tf.reduce_mean(loss)
        return self.__triplet_loss

    def __get_learning_rate_input(self):
        return self.__learning_rate_input

    def __get_momentum_input(self):
        return self.__momentum_input

    def set_learning_rate(self, new_learning_rate):
        sess = self.get_session()
        sess.run(fetches=[self.__assign_learning_rate], feed_dict={self.__get_learning_rate_input():
                                                                       new_learning_rate})

    def set_momentum(self, new_momentum):
        sess = self.get_session()
        sess.run(fetches=[self.__assign_momentum], feed_dict={self.__get_momentum_input():
                                                                  new_momentum})

    def __optimizer_to_use(self, loss_to_use):
        return tf.train.MomentumOptimizer(learning_rate=self.__learning_rate, momentum=self.__momentum). \
            minimize(loss=loss_to_use, var_list=self.__parameters)
        # return tf.train.RMSPropOptimizer(learning_rate=self.__learning_rate). \
        #    minimize(loss_to_use, var_list=self.__parameters)

    def __create_rms_triplet_loss(self):
        # forces the model to not learn the identity.
        if self.__triplet_loss is None:
            self.__create_triplet_loss()

        loss = tf.square(tf.subtract(self.__expected_output_data, self.__model))
        loss = tf.reduce_mean(loss)
        self.__rms_triplet_loss = tf.add(self.__triplet_loss, loss)
        self.__adam_rms_triplet = self.__optimizer_to_use(self.__rms_triplet_loss)

        # AdamOptimizer(learning_rate=self.__learning_rate) \
        #    .minimize(self.__rms_triplet_loss, var_list=self.__parameters)
        return self.__rms_triplet_loss

    def __create_rms_loss(self):
        loss = tf.square(tf.subtract(self.__expected_output_data, self.__model))
        loss = tf.reduce_mean(loss)
        self.__rms_loss = loss
        self.__adam_rms = self.__optimizer_to_use(self.__rms_loss)

        return self.__rms_loss

    @staticmethod
    def __psnr_loss_compute(expected_output, output):
        loss = tf.square(tf.subtract(expected_output, output))
        loss = tf.reduce_mean(loss)
        loss = tf.divide(1.0, loss)
        loss = tf.log(loss)
        loss = tf.divide(loss, tf.log(tf.constant(value=10.0, dtype=tf.float32)))
        loss = tf.multiply(loss, -10.0)
        return loss

    def __create_psnr_loss(self):
        self.__psnr_loss = SRModel.__psnr_loss_compute(self.__expected_output_data, self.__model)
        self.__adam_psnr_loss = self.__optimizer_to_use(self.__psnr_loss)

        x = tf.reduce_sum(self.__parameters[0])
        for i in range(1, len(self.__parameters)):
            x += tf.reduce_sum(self.__parameters[i] ** 2)
        return self.__psnr_loss + x

    def __create_psnr_triplet_loss(self):

        temp1 = tf.subtract(self.__input_data, self.__expected_output_data)
        temp2 = tf.subtract(self.__input_data, self.__model)
        temp1 = tf.square(temp1)
        temp2 = tf.square(temp2)

        loss = SRModel.__psnr_loss_compute(self.__expected_output_data, self.__model)

        triplet_loss = SRModel.__psnr_loss_compute(temp1, temp2)

        self.__psnr_triplet_loss = tf.add(triplet_loss, loss)
        self.__adam_psnr_triplet_loss = self.__optimizer_to_use(self.__psnr_triplet_loss)

        return self.__psnr_triplet_loss

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

    def __get_psnr_loss(self):
        if self.__psnr_loss is None:
            raise LossUninitialized
        return self.__psnr_loss

    def __get_psnr_triplet_loss(self):
        if self.__psnr_triplet_loss is None:
            raise LossUninitialized
        return self.__psnr_triplet_loss

    def __get_rms_loss_adam_opt(self):
        if self.__adam_rms is None:
            raise LossOptimizerUninitialized
        return self.__adam_rms

    def __get_rms_triplet_loss_adam_opt(self):
        if self.__adam_rms_triplet is None:
            raise LossOptimizerUninitialized
        return self.__adam_rms_triplet

    def __get_psnr_loss_adam_opt(self):
        if self.__adam_psnr_loss is None:
            raise LossOptimizerUninitialized
        return self.__adam_psnr_loss

    def __get_psnr_triplet_adam_opt(self):
        if self.__adam_psnr_triplet_loss is None:
            raise LossOptimizerUninitialized
        return self.__adam_psnr_triplet_loss

    def set_rms_triplet_loss(self):
        self.__active_loss = SRModel.RMS_TRIPLET_LOSS

    def set_rms_loss(self):
        self.__active_loss = SRModel.RMS_LOSS

    def set_psnr_loss(self):
        self.__active_loss = SRModel.PSNR_LOSS

    def set_psnr_triplet_loss(self):
        self.__active_loss = SRModel.PSNR_TRIPLET_LOSS

    def is_rms_triplet_loss_active(self):
        return self.__active_loss is SRModel.RMS_TRIPLET_LOSS

    def is_rms_loss_active(self):
        return self.__active_loss is SRModel.RMS_LOSS

    def is_psnr_loss_active(self):
        return self.__active_loss is SRModel.PSNR_LOSS

    def is_psnr_triplet_loss_active(self):
        return self.__active_loss is SRModel.PSNR_TRIPLET_LOSS

    def get_active_loss(self):
        return self.__active_loss

    def set_model_saver_inst(self, model_saver_inst):
        if not isinstance(model_saver_inst, model_saver.ModelSaver):
            raise InvalidType
        self.__model_saver_inst = model_saver_inst
        self.__model_saver_inst.load_checkpoint(sess=self.get_session())

    ORIGINAL_DIMENSION = SRDSManage.ORIGINAL_DIMENSION

    def prepare_train_test_dataset(self, train_ds_dir_list, test_ds_dir_list,
                                   num_of_training_ds_to_load, num_of_testing_ds_to_load,
                                   train_batch_size, test_batch_size=1,
                                   down_sample_train=4,
                                   down_sample_test=4,
                                   training_dimension=(100, 100), testing_dimension=ORIGINAL_DIMENSION,
                                   accepted_ext=DataSetManager.DEFAULT_IMG_EXTENSIONS):

        self.__ds_manage = SRDSManage(train_ds_dir_list, test_ds_dir_list,
                                      num_of_training_ds_to_load, num_of_testing_ds_to_load,
                                      down_sample_train=4,
                                      down_sample_test=4,
                                      training_dimension=training_dimension, testing_dimension=testing_dimension,
                                      accepted_ext=accepted_ext)

        self.__test_ds_container = self.__ds_manage.manufacture_test_batch_iterator(test_batch_size)
        self.__train_ds_container = self.__ds_manage.manufacture_train_batch_iterator(train_batch_size)

    def __get_active_optimizer(self):
        if self.is_rms_loss_active():
            optimizer = self.__get_rms_loss_adam_opt()
        elif self.is_rms_triplet_loss_active():
            optimizer = self.__get_rms_triplet_loss_adam_opt()
        elif self.is_psnr_loss_active():
            optimizer = self.__get_psnr_loss_adam_opt()
        elif self.is_psnr_triplet_loss_active():
            optimizer = self.__get_psnr_triplet_adam_opt()
        else:
            raise LossOptimizerUninitialized
        return optimizer

    def __get_active_loss(self):
        if self.is_rms_loss_active():
            loss = self.__get_rms_loss()
        elif self.is_rms_triplet_loss_active():
            loss = self.__get_rms_triplet_loss()
        elif self.is_psnr_loss_active():
            loss = self.__get_psnr_loss()
        elif self.is_psnr_triplet_loss_active():
            loss = self.__get_psnr_triplet_loss()
        else:
            raise LossUninitialized
        return loss

    def __run_get_loss(self, input_batch, output_batch, loss):
        sess = self.get_session()
        cur_train_loss = sess.run(fetches=[loss],
                                  feed_dict={self.__get_input_data_placeholder(): input_batch,
                                             self.__get_expected_output_placeholder(): output_batch})
        return cur_train_loss

    def __get_num_of_optimization_levels(self):
        return self.__num_of_optimization_levels

    def set_optimization_level(self, level):
        self.__num_of_optimization_levels = level

    def __run_optimizer(self, input_batch, output_batch, optimizer):
        # sess.run(fetches=[optimizer],
        #         feed_dict={self.__get_input_data_placeholder(): input_batch,
        #                    self.__get_expected_output_placeholder(): output_batch})

        level = self.__get_num_of_optimization_levels()

        sess = self.get_session()
        for i in range(level):
            __, input_batch = sess.run(fetches=[optimizer, self.get_model()],
                                       feed_dict={self.__get_input_data_placeholder(): input_batch,
                                                  self.__get_expected_output_placeholder(): output_batch})

        # img_result = sess.run(fetches=[self.get_model()],
        #                      feed_dict={self.__get_input_data_placeholder(): input_image})

    # @staticmethod
    # def run_train_default_args():

    # DEFAULT_CHECKPOINT_PRAMS = ModelSaver.checkpoint_model_default_args()

    def run_train(self, test_batch_size=1, train_batch_size=10, reinitialize_test_batch=False,
                  reinitialize_train_batch=False,
                  # parameters for train:
                  num_of_epochs=10,
                  display_status=True, display_status_iter=70, down_sample_factor=4,
                  running_avg_usage=True,
                  execute_tests=True,

                  # parameters for checkpoint:
                  save_checkpoints=True,
                  rebase_checkpoints=True):
        """
        :param rebase_checkpoints:
        :param reinitialize_train_batch:
        :param reinitialize_test_batch:
        :param train_batch_size:
        :param test_batch_size:
        :param running_avg_usage:
        :param execute_tests:
        :param save_checkpoints:
        :param num_of_epochs:
        :param display_status:
        :param display_status_iter:
        :param down_sample_factor:

        :return:
        """
        sess = self.get_session()
        ModelSaver = model_saver.ModelSaver
        if not isinstance(self.__ds_manage, SRDSManage) or \
                (not isinstance(self.__train_ds_container, BatchDataSetIterator) and reinitialize_train_batch):
            raise TrainDatasetNotInitialized

        if save_checkpoints and not isinstance(self.__model_saver_inst, model_saver.ModelSaver):
            raise SaveInstanceNotInitialized

        optimizer = self.__get_active_optimizer()
        train_loss = self.__get_active_loss()

        def print_checkpoint():
            print("Checkpoint committed")

        self.__model_saver_inst.set_first_run()

        running_avg = RunningAvg()
        if reinitialize_train_batch:
            self.__train_ds_container = self.__ds_manage.manufacture_train_batch_iterator(train_batch_size)

        iteration_count = 1
        total_iterations = num_of_epochs * len(self.__train_ds_container)
        cur_train_loss = [float('inf')]

        for epoch in range(num_of_epochs):

            single_epoch_iter = 0
            self.__train_ds_container.reset_batch_iterator()
            for io_train_datapoint in self.__train_ds_container:

                input_batch = DataBuffer.get_input_dict_dp(io_train_datapoint)
                output_batch = DataBuffer.get_output_dict_dp(io_train_datapoint)

                if running_avg_usage:
                    cur_train_loss = self.__run_get_loss(input_batch, output_batch, train_loss)
                    running_avg.add_to_avg(cur_train_loss[0])

                def get_train_loss():
                    return self.__run_get_loss(input_batch, output_batch, train_loss)

                if save_checkpoints:
                    self.__model_saver_inst.checkpoint_model_arguments \
                        (rebase_checkpoint=ModelSaver.REBASE_CHECKPOINT_IGNORE)

                    self.__model_saver_inst.checkpoint_model(checkpoint_loss=None,
                                                             exec_on_first_run=get_train_loss,
                                                             sess=sess)

                self.__run_optimizer(input_batch, output_batch, optimizer)

                if display_status and iteration_count % display_status_iter == 0:
                    if not running_avg_usage:
                        get_train_loss()
                    else:
                        cur_train_loss[0] = running_avg.get_avg()

                    print("Epoch: %d, iteration count: %d, epoch train percentage : %f%%, "
                          "total train percentage : %f%%,  training loss: %f" %
                          (epoch, iteration_count, ((float(single_epoch_iter) /
                                                     len(self.__train_ds_container)) * 100),
                           ((float(iteration_count) / total_iterations) * 100), cur_train_loss[0]))

                iteration_count += 1
                single_epoch_iter += 1

            cur_test_loss = self.run_test(down_sample_factor, display_status, test_batch_size,
                                          reinitialize_batches=reinitialize_test_batch)
            if save_checkpoints:
                def checkpoint_not_first_run():
                    raise CheckpointCannotBeFirstRun

                self.__model_saver_inst.force_checkpoint_model_execution()

                if rebase_checkpoints:
                    rebase = ModelSaver.REBASE_BEST_CHECKPOINT
                else:
                    rebase = ModelSaver.REBASE_CHECKPOINT_IGNORE
                self.__model_saver_inst.checkpoint_model_arguments \
                    (rebase_checkpoint=rebase)
                self.__model_saver_inst.checkpoint_model(checkpoint_loss=cur_test_loss[0],
                                                         exec_on_first_run=checkpoint_not_first_run,
                                                         sess=sess)
            if display_status:
                print("Epoch: %d, total train percentage : %f%%, test loss: %f" %
                      (epoch, ((float(iteration_count) / total_iterations) * 100), cur_test_loss[0]))

        if execute_tests:
            print("Final loss:")
            self.run_test(down_sample_factor=down_sample_factor,
                          display_status=display_status,
                          batch_size=test_batch_size,
                          reinitialize_batches=reinitialize_test_batch)

    def run_test(self, down_sample_factor=4,
                 display_status=True, batch_size=int(1), reinitialize_batches=False):
        """

        :param reinitialize_batches:
        :param batch_size:
        :param display_status:
        :param down_sample_factor:
        :param sess:
        :return:
        """

        if not isinstance(self.__ds_manage, SRDSManage) or \
                (not isinstance(self.__test_ds_container, BatchDataSetIterator) and reinitialize_batches):
            raise TestDatasetNotInitialized

        if not isinstance(self.__model_saver_inst, model_saver.ModelSaver):
            # used for reloading the trained parameters. It doesn't make sense to run this on random parameters.
            raise SaveInstanceNotInitialized

        test_loss = self.__get_active_loss()

        if reinitialize_batches:
            self.__test_ds_container = self.__ds_manage.manufacture_test_batch_iterator(batch_size)
        else:
            self.__test_ds_container.reset_batch_iterator()

        cur_test_loss = [0.0]
        count = 0

        for test_ds_iter in self.__test_ds_container:
            input_batch = DataBuffer.get_input_dict_dp(test_ds_iter)
            output_batch = DataBuffer.get_output_dict_dp(test_ds_iter)
            cur_test_loss[0] += self.__run_get_loss(input_batch, output_batch, test_loss)[0]
            count += 1

        cur_test_loss[0] = cur_test_loss[0] / count

        if display_status:
            print("Test loss: %f, Total batches checked : %d" % (cur_test_loss[0], count))

        return cur_test_loss

    def execute_model(self, input_image=None, input_image_path=None,
                      return_with_batch_column=True):

        if (input_image is None and input_image_path is None) or \
                (input_image is not None and input_image_path is not None):
            raise InvalidArgumentCombination

        if input_image_path is not None:
            input_image = cv2.imread(input_image_path)
            input_image = numpy.float32(input_image) / 255.0

        input_image = ensure_numpy_array(input_image)

        if len(input_image.shape) == 3:
            input_image = [input_image]

        sess = self.get_session()

        img_result = sess.run(fetches=[self.get_model()],
                              feed_dict={self.__get_input_data_placeholder(): input_image})

        if not return_with_batch_column:
            return ensure_numpy_array(img_result[0][0])

        return img_result[0]

    def get_model(self, **args):
        if self.__model is None:
            raise ModelNotInitialized
        return self.__model

    def get_parameter_tensors(self):
        if self.__parameters is None:
            raise ParameterNotInitialized

        return self.__parameters


if __name__ == "__main__":
    def main_fnc():
        input()

        model_instance = SRModel()
        parameters = model_instance.get_parameter_tensors()

        modelsave = model_saver.ModelSaver(
            'exp1_', parameters,
            save_file_path="/media/sreramk/storage-main/elementary_frame/checkpoints/")

        model_instance.set_model_saver_inst(modelsave)

        # model_instance.prepare_train_test_dataset(
        #    ['/media/sreramk/storage-main/elementary_frame/dataset/DIV2K_train_HR/'],
        #    ['/media/sreramk/storage-main/elementary_frame/dataset/DIV2K_valid_HR/'],
        #    num_of_training_ds_to_load=160, num_of_testing_ds_to_load=40,
        #    train_batch_size=10, testing_dimension=(500, 500)
        # )

        model_instance.set_psnr_loss()
        # model_instance.run_test()
        # print("active loss: " + model_instance.get_active_loss())

        img = model_instance.fetch_image(  # '/media/sreramk/storage-main/elementary_frame/dataset/t1.png',
            '/media/sreramk/storage-main/elementary_frame/dataset/DIV2K_valid_HR/0820.png',
            with_batch_column=False)
        # model_instance.display_image(img)

        modelsave.checkpoint_model_arguments(skip_duration=100)
        model_instance.set_learning_rate(0.1)
        # model_instance.run_train(num_of_epochs=100)
        _, __, img = DataSetManager.random_crop_img(img, 250, 250)
        img = model_instance.zoom_image(img, 4, 4)
        for i in range(20):
            cv2.imshow("im1", img)
            # model_instance.display_image(img)
            result = model_instance.execute_model(input_image=img, return_with_batch_column=False)
            # print(result)
            # model_instance.display_image(result)
            cv2.imshow("img2", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            img = result

            # _, __, img = DataSetManager.random_crop_img(img, 250, 250)
            # img = model_instance.zoom_image(img, 4, 4)

        img = model_instance.fetch_image(
            "https://cdn.insidetheperimeter.ca/wp-content/uploads/2015/11/Albert_einstein_by_zuzahin-d5pcbug-WikiCommons-768x706.jpg")
        size_x, size_y = model_instance.get_image_dimensions(img)


    main_fnc()
