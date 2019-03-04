# copyright (c) 2019 K Sreram, All rights reserved.
from os import listdir
from os.path import isfile, join
import urllib.parse
import cv2
import os
import matplotlib.pyplot as plt
from skimage import io
import numpy

from prepare_dataset.sr_image_ds_manager import ImageDSManage
from trainer.exceptions import LossUninitialized, LossOptimizerUninitialized, RequiredPlaceHolderNotInitialized, \
    TrainDatasetNotInitialized, TestDatasetNotInitialized, SaveInstanceNotInitialized, RunTestError, \
    ModelNotInitialized, ParameterNotInitialized, InvalidArgumentCombination, InvalidType
from trainer.model_base import ModelBase
import tensorflow as tf
from model_saver_manager import model_saver


class SRModel(ModelBase):
    RMS_TRIPLET_LOSS = "rms_triplet_loss"
    RMS_LOSS = "rms_loss"

    def __init__(self):
        self.__input_data = None
        self.__expected_output_data = None

        self.__train_ds_manage = None
        self.__test_ds_manage = None

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

    def prepare_train_test_dataset(self, train_dataset_path, test_dataset_path, down_sample_factor,
                                   image_buffer_limit_train=int(100 * 0.8), image_buffer_limit_test=int(100 * 0.2),
                                   buffer_priority=1000):
        self.__train_ds_manage = ImageDSManage(train_dataset_path,
                                               image_buffer_limit=image_buffer_limit_train,
                                               buffer_priority=buffer_priority)

        self.__test_ds_manage = ImageDSManage(test_dataset_path,
                                              image_buffer_limit=image_buffer_limit_test,
                                              buffer_priority=buffer_priority)

        self.__train_ds_manage.fill_buffer(down_sample_factor)
        self.__test_ds_manage.fill_buffer(down_sample_factor)

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

    def run_train(self,
                  # parameters for train:
                  num_of_epochs=10, single_epoch_count=12010, checkpoint_iteration_count=3000,
                  display_status=True, display_status_iter=100, batch_size=10, down_sample_factor=4,
                  min_x_f=100, min_y_f=100,

                  # parameters for test:
                  terminal_loss=0.00001, test_min_x_f=500, test_min_y_f=500, number_of_samples=int(100),
                  run_test_periodically=3000, break_samples_by=int(10),
                  execute_tests=True,

                  # parameters for checkpoint:
                  save_checkpoints=True):
        """

        :param break_samples_by:
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

        optimizer = self.__get_active_optimizer()
        train_loss = self.__get_active_loss()

        def print_checkpoint():
            print("Checkpoint committed")

        self.__model_saver_inst.set_first_run()

        def execute_all_epoch():
            with tf.Session() as sess:
                init = tf.initialize_all_variables()
                sess.run(init)
                iteration_count = 0
                total_iterations = num_of_epochs * single_epoch_count

                def get_train_loss():
                    result_ = self.__train_ds_manage.get_batch(batch_size=batch_size,
                                                               down_sample_factor=down_sample_factor,
                                                               min_x_f=min_x_f,
                                                               min_y_f=min_y_f)

                    min_x_, min_y_, batch_down_sampled_, batch_original_ = result_

                    return sess.run(fetches=[train_loss],
                                    feed_dict={self.__get_input_data_placeholder(): batch_down_sampled_,
                                               self.__get_expected_output_placeholder(): batch_original_})

                for epoch in range(num_of_epochs):

                    for i in range(single_epoch_count):

                        if save_checkpoints:
                            cur_train_loss = get_train_loss()
                            self.__model_saver_inst.checkpoint_model(float(cur_train_loss[0]), sess=sess,
                                                                     skip_type=ModelSaver.TimeIterSkipManager.ST_ITER_SKIP,
                                                                     skip_duration=checkpoint_iteration_count,
                                                                     exec_on_checkpoint=print_checkpoint)

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
                            print("Epoch: %d, iteration count: %d, epoch train percentage : %.3f%%, "
                                  "total train percentage : %.3f%%,  training loss: %f" %
                                  (epoch, iteration_count, ((float(i) / single_epoch_count) * 100),
                                   ((float(iteration_count) / total_iterations) * 100), cur_train_loss[0]))

                        if execute_tests and iteration_count % run_test_periodically == 0:
                            cur_test_loss = \
                                self.run_test(test_min_x_f, test_min_y_f, number_of_samples, sess, down_sample_factor,
                                              display_status, break_samples_by)

                            if cur_test_loss[0] < terminal_loss:
                                return

                        iteration_count += 1

                    # display at the end of epoch, if not displayed
                    if display_status and iteration_count % display_status_iter != 0:
                        cur_train_loss = get_train_loss()
                        print("Epoch: %d, total train percentage : %.3f%%, training loss: %f" %
                              (epoch, ((float(iteration_count) / total_iterations) * 100), cur_train_loss[0]))

        execute_all_epoch()
        if execute_tests:
            print("Final loss:")
            with tf.Session() as sess:
                self.run_test(test_min_x_f, test_min_y_f, number_of_samples, sess, down_sample_factor,
                              display_status, break_samples_by)

    def run_test(self, min_x_f=500, min_y_f=500, number_of_samples=int(50 * 0.2), sess=None, down_sample_factor=4,
                 display_status=True, break_samples_by=int(50 * 0.2)):
        """

        :param fill_ds_buffer:
        :param break_samples_by:
        :param display_status:
        :param down_sample_factor:
        :param sess:
        :param min_x_f:
        :param min_y_f:
        :param number_of_samples:
        :return:
        """
        if not isinstance(self.__test_ds_manage, ImageDSManage):
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

        def get_test_loss(batch_size):
            result = self.__test_ds_manage.get_batch(batch_size=batch_size,
                                                     down_sample_factor=down_sample_factor,
                                                     min_x_f=min_x_f,
                                                     min_y_f=min_y_f)

            min_x, min_y, batch_down_sampled, batch_original = result
            return sess.run(fetches=[test_loss],
                            feed_dict={self.__get_input_data_placeholder(): batch_down_sampled,
                                       self.__get_expected_output_placeholder(): batch_original})

        cur_test_loss = [0.0]
        count = 0
        temp_number_of_samples = number_of_samples
        if temp_number_of_samples > break_samples_by:
            while temp_number_of_samples >= break_samples_by:
                cur_test_loss[0] += get_test_loss(break_samples_by)[0]
                count += 1
                temp_number_of_samples -= break_samples_by
            cur_test_loss[0] = cur_test_loss[0] / count
        else:
            cur_test_loss = get_test_loss(temp_number_of_samples)

        if display_status:
            print("Test loss: %.3f, Total samples checked : %d" % (cur_test_loss[0], number_of_samples))

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
            save_file_path="/media/sreramk/storage-main/elementary_frame/model_checkpoints_new/")

        model_instance.set_model_saver_inst(modelsave)

        model_instance.prepare_train_test_dataset(
            ['/media/sreramk/storage-main/elementary_frame/dataset/DIV2K_train_HR/'],
            ['/media/sreramk/storage-main/elementary_frame/dataset/DIV2K_valid_HR/'],
            down_sample_factor=4
        )

        model_instance.set_rms_loss()
        model_instance.run_test(number_of_samples=32)
        print("active loss: " + model_instance.get_active_loss())

        img = model_instance.fetch_image('/media/sreramk/storage-main/elementary_frame/dataset/DIV2K_valid_HR/0848.png',
                                         with_batch_column=False)
        # model_instance.display_image(img)
        model_instance.run_train(num_of_epochs=1, single_epoch_count=3010)
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

        img = model_instance.fetch_image("https://cdn.insidetheperimeter.ca/wp-content/uploads/2015/11/Albert_einstein_by_zuzahin-d5pcbug-WikiCommons-768x706.jpg")
        size_x, size_y = model_instance.get_image_dimensions(img)



    main_fnc()
