# copyright (c) 2019 K Sreram, All rights reserved.
import h5py
import numpy

from utils import RandomDict
from h5py_wrappers import H5PyDict


class H5PyRandDict(RandomDict):
    @staticmethod
    def prepare_path(cur_path):
        if cur_path[len(cur_path) - 1] != "/":
            return cur_path + "/"
        return cur_path

    class H5PyDictIterator:

        def __init__(self, keys):
            self.__keys = keys
            self.__keys_iter = iter(keys)

        def __next__(self):
            return next(self.__keys_iter)

    def __init__(self, storage_inst:h5py.Group, seq=None, **kwargs):

        self.__storage_inst = storage_inst

        self.__storage_key_to_val = H5PyDict(self.__storage_inst.require_group(RandomDict.KEY_TO_VAL + "/"))
        self.__storage_key_to_sno = H5PyDict(self.__storage_inst.require_group(RandomDict.KEY_TO_SNO + "/"))
        self.__storage_sno_to_key = H5PyDict(self.__storage_inst.require_group(RandomDict.SNO_TO_KEY + "/"))

        super().__init__(seq, inst_key_to_sno=self.__storage_key_to_sno,
                         inst_sno_to_key=self.__storage_sno_to_key,
                         inst_key_to_val=self.__storage_key_to_val, **kwargs)

    def create_memory_friendly_copy(self, working_dir=None):
        return H5PyRandDict(storage_inst=self.__storage_inst.require_group("temp/"))

    @staticmethod
    def prepare_key(key):
        if isinstance(key, int):
            return str(key)

    def __getitem__(self, key):
        key = H5PyRandDict.prepare_key(key)
        return numpy.array(super()[key])

    def __setitem__(self, key, value):

        key = H5PyRandDict.prepare_key(key)

        self.__storage_sno_to_key.get_working_mode().set_create_working_mode()
        self.__storage_key_to_sno.get_working_mode().set_create_working_mode()
        self.__storage_key_to_val.get_working_mode().set_create_replace_working_mode()

        super().__setitem__(key, value)

    def __delitem__(self, key):
        key = H5PyRandDict.prepare_key(key)
        self.__storage_key_to_sno.get_working_mode().set_replace_working_mode()
        self.__storage_sno_to_key.get_working_mode().set_replace_working_mode()
        self.__storage_key_to_val.get_working_mode().set_replace_working_mode()

        super().__delitem__(key)
