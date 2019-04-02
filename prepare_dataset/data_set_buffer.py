#  copyright (c) 2019 K Sreram, All rights reserved.
from enum import Enum

import collections

from prepare_dataset.exceptions import InvalidFlag, BufferOverflow, ImageNotFound, BufferIsEmpty, \
    UniqueGetIsNotInitialized, InvalidDataPointDefinition
from utils import RandomDict
from utils.exceptions import KeyDoesNotExist
from utils.general_utils import ensure_numpy_array


class DataBuffer:
    SIZE_UNLIMITED = -1
    SAME = -1

    class COMPARE(Enum):
        GREATER = 'greater'
        LESSER = 'lesser'
        EQUAL = 'equal'

    def __init__(self, buffer_max_size=SIZE_UNLIMITED, rnd_dict_inst=None):
        if rnd_dict_inst is None:
            rnd_dict_inst = RandomDict()
        self.__rand_query_store = rnd_dict_inst  # dictionary that allows a random key to be returned in O(1) time
        self.__buffer_max_size = buffer_max_size

        self.__rand_query_store_unique = None

    @staticmethod
    def __comp_buffer_size(buffer_storage, size):
        if size < 0:
            # represents a flag
            if size == DataBuffer.SIZE_UNLIMITED:
                return DataBuffer.COMPARE.LESSER
            else:
                raise InvalidFlag

        if len(buffer_storage) > size:
            return DataBuffer.COMPARE.GREATER
        elif len(buffer_storage) < size:
            return DataBuffer.COMPARE.LESSER
        else:
            return DataBuffer.COMPARE.EQUAL

    @staticmethod
    def create_input_output_dp(input_dp, output_dp):
        return [ input_dp, output_dp]

    @staticmethod
    def get_input_dict_dp(io_dp):
        return ensure_numpy_array(io_dp[0])

    @staticmethod
    def get_output_dict_dp(io_dp):
        return ensure_numpy_array(io_dp[1])

    def is_buffer_full(self):
        compare = DataBuffer.__comp_buffer_size(self.__rand_query_store, self.__buffer_max_size)
        return (compare == DataBuffer.COMPARE.GREATER) or (compare == DataBuffer.COMPARE.EQUAL)

    def is_buffer_empty(self):
        return DataBuffer.__comp_buffer_size(self.__rand_query_store, 0) == DataBuffer.COMPARE.EQUAL

    def add_to_buffer(self, key, data_point):
        """
        Adds elements to the random_query_storage
        :param key:
        :param data_point:
        :return:
        """
        # img = general_utils.ensure_numpy_array(img)

        # DataBuffer.__exception_on_non_dp(data_point)

        if self.is_buffer_full():
            raise BufferOverflow

        self.__rand_query_store.add_element(key, data_point)

    def clear_buffer(self, buffer_max_size=SAME):

        if buffer_max_size != DataBuffer.SAME:
            self.__buffer_max_size = buffer_max_size

        self.__rand_query_store.clear()

    def get_data(self, key):
        try:
            return self.__rand_query_store.get_element(key)
        except KeyDoesNotExist:
            raise ImageNotFound

    def get_random(self):

        if self.is_buffer_empty():
            raise BufferIsEmpty

        self.__rand_query_store.random_value()

    def reset_get_random_unique(self):
        self.__rand_query_store_unique = None

    def initialize_get_random_unique(self):

        self.__rand_query_store_unique = self.__rand_query_store.create_memory_friendly_copy()

    def get_random_unique(self):
        if self.__rand_query_store_unique is None:
            raise UniqueGetIsNotInitialized

        if len(self.__rand_query_store_unique) == 0:
            self.reset_get_random_unique()
            return None

        key, img = self.__rand_query_store_unique.random_key_value()

        self.__rand_query_store_unique.delete_element(del_key=key)

        return img
