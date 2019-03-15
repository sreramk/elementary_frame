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
from prepare_dataset.sr_image_ds_manager import ImageDSManage
from trainer.exceptions import LossUninitialized, LossOptimizerUninitialized, RequiredPlaceHolderNotInitialized, \
    TrainDatasetNotInitialized, TestDatasetNotInitialized, SaveInstanceNotInitialized, ModelNotInitialized, \
    ParameterNotInitialized, InvalidArgumentCombination, InvalidType
from trainer.model_base import ModelBase
from utils.running_avg import RunningAvg


class SRModel(ModelBase):
    RMS_TRIPLET_LOSS = "rms_triplet_loss"
    RMS_LOSS = "rms_loss"

    def __init__(self):
        self.__input_data = None
        self.__expected_output_data = None

        self.__test_ds_container = None
        self.__train_ds_container = None

        self.__ds_manage = None

        self.__adam_rms = None

        self.__adam_rms_triplet = None

        self.__model_saver_inst = None

        self.__filters = []

        self.__biases = []

        self.__model = None

        self.__triplet_loss = None

        self.__rms_triplet_loss = None

        self.__rms_loss = None

        self.__active_loss = SRModel.RMS_TRIPLET_LOSS

        self.__padding = "SAME"

        self.__strides = [1, 1, 1, 1]

        self.__create_place_holders()

        self.__create_parameters()

        self.__create_model()

        self.__create_triplet_loss()

        self.__create_rms_loss()

        self.__create_rms_triplet_loss()

        super().__init__()

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

        input_image = ImageDSManage.ensure_numpy_array(input_image)

        if len(input_image.shape) == 3 and with_batch_column:
            input_image = [input_image]

        input_image = ImageDSManage.ensure_numpy_array(input_image)

        return input_image

    @staticmethod
    def get_image_dimensions(img):
        if ImageDSManage.check_if_numpy_array(img):
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
        return ImageDSManage.ensure_numpy_array(result)

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

        self.__parameters = list(self.__filters)
        self.__parameters.extend(self.__biases)

        return self.__filters, self.__biases, self.__parameters

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
        self.__active_loss = SRModel.RMS_TRIPLET_LOSS

    def set_rms_loss(self):
        self.__active_loss = SRModel.RMS_LOSS

    def is_rms_triplet_loss_active(self):
        return self.__active_loss == SRModel.RMS_TRIPLET_LOSS

    def is_rms_loss_active(self):
        return self.__active_loss == SRModel.RMS_LOSS

    def get_active_loss(self):
        return self.__active_loss

    def set_model_saver_inst(self, model_saver_inst):
        if not isinstance(model_saver_inst, model_saver.ModelSaver):
            raise InvalidType
        self.__model_saver_inst = model_saver_inst

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
        else:
            optimizer = self.__get_rms_triplet_loss_adam_opt()

        return optimizer

    def __get_active_loss(self):
        if self.is_rms_loss_active():
            loss = self.__get_rms_loss()
        else:
            loss = self.__get_rms_triplet_loss()
        return loss

    def __run_get_loss(self, sess, input_batch, output_batch, loss):

        cur_train_loss = sess.run(fetches=[loss],
                                     feed_dict={self.__get_input_data_placeholder(): input_batch,
                                                self.__get_expected_output_placeholder(): output_batch})
        return cur_train_loss

    def __run_optimizer(self, sess, input_batch, output_batch, optimizer):
        sess.run(fetches=[optimizer],
                 feed_dict={self.__get_input_data_placeholder(): input_batch,
                            self.__get_expected_output_placeholder(): output_batch})

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
                  save_checkpoints=True):
        """
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

        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)
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
                        cur_train_loss = self.__run_get_loss(sess, input_batch, output_batch, train_loss)
                        running_avg.add_to_avg(cur_train_loss[0])

                    def get_train_loss():
                        return self.__run_get_loss(sess, input_batch, output_batch, train_loss)

                    if save_checkpoints:
                        self.__model_saver_inst.checkpoint_model_arguments\
                            (rebase_checkpoint=ModelSaver.REBASE_CHECKPOINT_IGNORE)

                        self.__model_saver_inst.checkpoint_model(checkpoint_efficiency=None,
                                                                 exec_on_first_run=get_train_loss,
                                                                 sess=sess)

                    self.__run_optimizer(sess, input_batch, output_batch, optimizer)

                    if display_status and display_status_iter % iteration_count == 0:
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

                cur_test_loss = self.run_test(sess, down_sample_factor, display_status, test_batch_size,
                                              reinitialize_batches=reinitialize_test_batch)
                if save_checkpoints:

                    def checkpoint_not_first_run():
                        raise CheckpointCannotBeFirstRun

                    self.__model_saver_inst.force_checkpoint_model_execution()
                    self.__model_saver_inst.checkpoint_model_arguments \
                        (rebase_checkpoint=ModelSaver.REBASE_BEST_CHECKPOINT)
                    self.__model_saver_inst.checkpoint_model(checkpoint_efficiency=cur_test_loss[0],
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

    def run_test(self, sess=None, down_sample_factor=4,
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

        session_is_parent = True if sess is not None else False

        if not session_is_parent:
            sess = tf.Session()
            init = tf.initialize_all_variables()
            sess.run(init)

        # loads the most recently used checkpoint.
        self.__model_saver_inst.load_checkpoint(sess=sess)
        if reinitialize_batches:
            self.__test_ds_container = self.__ds_manage.manufacture_test_batch_iterator(batch_size)
        else:
            self.__test_ds_container.reset_batch_iterator()

        cur_test_loss = [0.0]
        count = 0

        for test_ds_iter in self.__test_ds_container:
            input_batch = DataBuffer.get_input_dict_dp(test_ds_iter)
            output_batch = DataBuffer.get_output_dict_dp(test_ds_iter)
            cur_test_loss[0] += self.__run_get_loss(sess, input_batch, output_batch, test_loss)[0]
            count += 1

        cur_test_loss[0] = cur_test_loss[0] / count

        if display_status:
            print("Test loss: %f, Total batches checked : %d" % (cur_test_loss[0], count))

        if not session_is_parent:
            sess.close()

        return cur_test_loss

    def execute_model(self, input_image=None, input_image_path=None, sess=None,
                      return_with_batch_column=True):

        if (input_image is None and input_image_path is None) or \
                (input_image is not None and input_image_path is not None):
            raise InvalidArgumentCombination

        if input_image_path is not None:
            input_image = cv2.imread(input_image_path)
            input_image = numpy.float32(input_image) / 255.0

        input_image = ImageDSManage.ensure_numpy_array(input_image)

        if len(input_image.shape) == 3:
            input_image = [input_image]

        session_is_parent = True if sess is not None else False

        if not session_is_parent:
            sess = tf.Session()
            init = tf.initialize_all_variables()
            sess.run(init)

        # loads the most recently used checkpoint.
        self.__model_saver_inst.load_checkpoint(sess=sess)

        img_result = sess.run(fetches=[self.get_model()],
                              feed_dict={self.__get_input_data_placeholder(): input_image})

        if not session_is_parent:
            sess.close()

        if not return_with_batch_column:
            return ImageDSManage.ensure_numpy_array(img_result[0][0])

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

        model_instance.prepare_train_test_dataset(
            ['/media/sreramk/storage-main/elementary_frame/dataset/DIV2K_train_HR/'],
            ['/media/sreramk/storage-main/elementary_frame/dataset/DIV2K_valid_HR/'],
            num_of_training_ds_to_load=160, num_of_testing_ds_to_load=40,
            train_batch_size=10, testing_dimension=(500, 500)
        )

        model_instance.set_rms_loss()
        model_instance.run_test()
        print("active loss: " + model_instance.get_active_loss())

        img = model_instance.fetch_image('/media/sreramk/storage-main/elementary_frame/dataset/DIV2K_valid_HR/0848.png',
                                         with_batch_column=False)
        # model_instance.display_image(img)
        model_instance.run_train(num_of_epochs=100)
        for i in range(3):
            _, __, img = ImageDSManage.random_crop(img, 250, 250)
            img = model_instance.zoom_image(img, 4, 4)
            cv2.imshow("im1", img)
            # model_instance.display_image(img)
            result = model_instance.execute_model(input_image=img, return_with_batch_column=False)
            # print(result)
            # model_instance.display_image(result)
            cv2.imshow("img2", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            img = result

        img = model_instance.fetch_image(
            "https://cdn.insidetheperimeter.ca/wp-content/uploads/2015/11/Albert_einstein_by_zuzahin-d5pcbug-WikiCommons-768x706.jpg")
        size_x, size_y = model_instance.get_image_dimensions(img)


    main_fnc()
