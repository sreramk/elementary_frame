# copyright (c) 2019 K Sreram, all rights reserved.
import random

import cv2
import numpy

from prepare_dataset.data_set_buffer import DataBuffer
from prepare_dataset.dataset_manager import DataSetManager
from prepare_dataset.exceptions import InvalidDataSetLabel, StopPopulatingTrainOrTestBuffer
from utils.general_utils import ensure_numpy_array


class SRDSManage(DataSetManager):

    DEFAULT_TRAIN_DOWN_SAMPLE = 4
    DEFAULT_TEST_DOWN_SAMPLE = 4
    ORIGINAL_DIMENSION = [-1, -1]

    def __init__(self, train_ds_dir_list, test_ds_dir_list,
                 num_of_training_ds_to_load, num_of_testing_ds_to_load,
                 down_sample_train=DEFAULT_TRAIN_DOWN_SAMPLE,
                 down_sample_test=DEFAULT_TEST_DOWN_SAMPLE,
                 training_dimension=ORIGINAL_DIMENSION, testing_dimension=ORIGINAL_DIMENSION,
                 accepted_ext=DataSetManager.DEFAULT_IMG_EXTENSIONS
                 ):
        self.__training_dimension = training_dimension
        self.__testing_dimension = testing_dimension

        self.__train_down_sample = down_sample_train

        self.__test_down_sample = down_sample_test

        super().__init__(train_ds_dir_list, test_ds_dir_list,
                         num_of_training_ds_to_load, num_of_testing_ds_to_load,
                         accepted_ext=accepted_ext)

    def set_train_downsample(self, new_frac):
        self.__train_down_sample = new_frac

    def get_train_downsample(self):
        return self.__train_down_sample

    def set_test_downsample(self, new_frac):
        self.__test_down_sample = new_frac

    def get_test_downsample(self):
        return self.__test_down_sample

    def _reset_get(self):
        """
        This will only be required if the get method acquires random data.
        :return:
        """
        pass

    def _preprocessor(self, data, ds_label):
        """
        This returns an instance of a fully prepared data-point
        :param data:
        :return:
        """

        if ds_label == DataSetManager.DsType.TRAINING:
            down_sample_fact = self.get_train_downsample()
            dimensions = self.__training_dimension
        elif ds_label == DataSetManager.DsType.TESTING:
            down_sample_fact = self.get_test_downsample()
            dimensions = self.__testing_dimension
        else:
            raise InvalidDataSetLabel

        if dimensions != SRDSManage.ORIGINAL_DIMENSION:
            _, __, data = DataSetManager.random_crop_img(data, dimensions[0],
                                                         dimensions[1])

        down_sampled = cv2.resize(data, dsize=(int(len(data[0]) / down_sample_fact),
                                               int(len(data) / down_sample_fact)), interpolation=cv2.INTER_NEAREST)

        down_sampled = cv2.resize(down_sampled, dsize=(len(data[0]), len(data)),
                                  interpolation=cv2.INTER_NEAREST)

        # down_sampled = cv2.GaussianBlur(down_sampled, (3, 3), 0)

        data = ensure_numpy_array(data)
        # data = 1-data
        # data = numpy.flip(data, axis=0)
        down_sampled = ensure_numpy_array(down_sampled)

        return DataBuffer.create_input_output_dp(down_sampled, data)

    def _get_data(self, ds_label):
        directory = None

        try:
            if ds_label == DataSetManager.DsType.TRAINING:
                directory = self._get_train_dir_rand()
            elif ds_label == DataSetManager.DsType.TESTING:
                directory = self._get_test_dir_rand()
        except IndexError:
            raise StopPopulatingTrainOrTestBuffer

        if directory is None:
            raise InvalidDataSetLabel

        new_img = cv2.imread(directory)
        new_img = numpy.float32(new_img) / 255.0
        new_img = ensure_numpy_array(new_img)
        return self._preprocessor(new_img, ds_label)

    def _get_train_dir_rand(self):
        return random.choice(self.get_train_ds_dir_list())

    def _get_test_dir_rand(self):
        return random.choice(self.get_test_ds_dir_list())