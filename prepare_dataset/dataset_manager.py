# Copyright (c) 2019 K Sreram, All rights reserved
import h5py
import os
import random
from enum import Enum

from prepare_dataset.batch_dataset_iterator import BatchDataSetIterator
from prepare_dataset.data_set_buffer import DataBuffer
from prepare_dataset.exceptions import MethodNotOverridden, StopPopulatingTrainOrTestBuffer
from utils.general_utils import ensure_numpy_array


class DataSetManager:
    DEFAULT_IMG_EXTENSIONS = [".jpeg", ".jpg", ".png", ".bmp"]

    class DsType(Enum):
        TRAINING = "training"
        TESTING = "testing"

    class H5PyDBPaths(Enum):
        TRAIN_BUFFER = "train_buffer"
        TEST_BUFFER = "test_buffer"

    @staticmethod
    def get_all_files_in_dir_list(dir_list, accepted_ext):
        result = []

        for s in dir_list:
            temp_dirs = list(os.listdir(s))

            for i in range(len(temp_dirs)):
                temp_dirs[i] = s + temp_dirs[i]

            new_dirs = []

            for strings in temp_dirs:
                for extInst in accepted_ext:
                    if extInst in strings[len(strings) - len(extInst):]:
                        new_dirs.append(strings)

            result.extend(list(new_dirs))

        return result

    @staticmethod
    def crop_img(image, size_x, size_y, pos_x, pos_y):
        start_x = pos_x
        start_y = pos_y
        end_x = size_x + start_x
        end_y = size_y + start_y
        return ensure_numpy_array(image[start_y:end_y, start_x:end_x])

    @staticmethod
    def random_crop_img(image, size_x, size_y):

        delta_x = len(image[0]) - size_x
        start_x = 0
        if delta_x != 0:
            start_x = random.randrange(0, delta_x)

        delta_y = len(image) - size_y
        start_y = 0
        if delta_y != 0:
            start_y = random.randrange(0, delta_y)

        end_x = size_x + start_x
        end_y = size_y + start_y

        return start_x, start_y, ensure_numpy_array(image[start_y:end_y, start_x:end_x])

    def __init__(self, train_ds_dir_list, test_ds_dir_list,
                 num_of_training_ds_to_load, num_of_testing_ds_to_load,
                 accepted_ext, hy5_group_inst=None):

        self.__train_ds_dir_list = train_ds_dir_list
        self.__test_ds_dir_list = test_ds_dir_list

        self.__train_dir_list = DataSetManager.get_all_files_in_dir_list(self.__train_ds_dir_list,
                                                                         accepted_ext)
        self.__test_dir_list = DataSetManager.get_all_files_in_dir_list(self.__test_ds_dir_list,
                                                                        accepted_ext)

        self.__num_of_training_ds_to_load = num_of_training_ds_to_load

        self.__num_of_testing_ds_to_load = num_of_testing_ds_to_load

        self.__train_buffer = None
        self.__test_buffer = None

        self.__train_buffer_db_inst = None
        self.__test_buffer_db_inst = None

        self.__h5py_group_inst = hy5_group_inst
        self.__initialize_dataset()

    def _reset_get(self):
        """
        This must reset the get procedure, which helps populate the buffer.
        :return:
        """
        raise MethodNotOverridden

    def _preprocessor(self, data, ds_label):
        """
        This returns an instance of a fully prepared data-point which includes the input and the output in a list
        :param data:
        :return:
        """
        raise MethodNotOverridden

    def _get_data(self, ds_label):
        """
        This method gets the data-set. This is repeatedly called for multiple times, to get the entire data-set.
        :param ds_label:
        :return:
        """
        raise MethodNotOverridden

    def __initialize_dataset(self):

        if self.__h5py_group_inst is not None:
            self.__h5py_group_inst: h5py.Group
            self.__train_buffer_db_inst = \
                self.__h5py_group_inst.require_group(DataSetManager.H5PyDBPaths.TRAIN_BUFFER)
            self.__test_buffer_db_inst = \
                self.__h5py_group_inst.require_group(DataSetManager.H5PyDBPaths.TEST_BUFFER)

        self.__train_buffer = DataBuffer(rnd_dict_inst=self.__train_buffer_db_inst)
        self.__test_buffer = DataBuffer(rnd_dict_inst=self.__test_buffer_db_inst)
        self._reset_get()

        try:
            for count in range(self.__num_of_training_ds_to_load):
                cur_data = self._get_data(DataSetManager.DsType.TRAINING)
                # DataSetManager.__exception_on_non_dp(cur_data)
                self.__train_buffer.add_to_buffer(count, cur_data)
        except StopPopulatingTrainOrTestBuffer:
            # nothing have to be done here, this helps break the loop when there is no more available data. This method
            # is tolerant towards getting less data than promised
            pass

        try:
            for count in range(self.__num_of_testing_ds_to_load):
                cur_data = self._get_data(DataSetManager.DsType.TESTING)
                # DataSetManager.__exception_on_non_dp(cur_data)
                self.__test_buffer.add_to_buffer(count, cur_data)
        except StopPopulatingTrainOrTestBuffer:
            # nothing have to be done here, this helps break the loop when there is no more available data. This method
            # is tolerant towards getting less data than promised
            pass

    def get_train_ds_dir_list(self):
        return self.__train_dir_list

    def get_test_ds_dir_list(self):
        return self.__test_dir_list

    def manufacture_train_batch_iterator(self, batch_size):
        train_batches = BatchDataSetIterator(self.__train_buffer, batch_size)
        return train_batches

    def manufacture_test_batch_iterator(self, batch_size):
        test_batches = BatchDataSetIterator(self.__test_buffer, batch_size)
        return test_batches
