# copyright (c) 2019 K Sreram, All rights reserved.

import numpy

from utils import RandomDict
from utils.exceptions import InitializationError
from utils.h5py_dict import H5PyDict


class H5PyRandDict(RandomDict):
    MAIN_WORKING_GROUP = "/main/"

    TEMP_WORKING_GROUP = "temp/"

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

    def set_data_path(self, name):
        return self.__h5py_group_path + name + "/"

    def __init__(self, storage_inst=None, group_inst=None,
                 database_name=None, working_dir=None, seq=None, h5py_group_path=MAIN_WORKING_GROUP, **kwargs):

        self.__database_name = database_name
        self.__working_dir = working_dir
        self.__h5py_group_path = h5py_group_path

        if self.__h5py_group_path is not None:
            key_to_sno_dbpath = self.__h5py_group_path + RandomDict.KEY_TO_SNO + "/"
            sno_to_key_dbpath = self.__h5py_group_path + RandomDict.SNO_TO_KEY + "/"
            key_to_val_dbpath = self.__h5py_group_path + RandomDict.KEY_TO_VAL + "/"
        else:
            key_to_sno_dbpath, sno_to_key_dbpath, key_to_val_dbpath = \
                RandomDict.KEY_TO_SNO+"/", RandomDict.SNO_TO_KEY+"/", RandomDict.KEY_TO_VAL+"/"

        self.__group_inst = group_inst
        self.__h5py_store = storage_inst

        try:
            self.__storage_key_to_sno = H5PyDict(working_dir=self.__working_dir, database_name=self.__database_name,
                                             h5py_group_path=key_to_sno_dbpath)
            self.__h5py_store = self.__storage_key_to_sno.get_h5py_storage_inst()
        except InitializationError:
            self.__storage_key_to_sno = H5PyDict(h5py_storage_inst=self.__h5py_store, group_inst=self.__group_inst,
                                                 h5py_group_path=key_to_sno_dbpath)

        self.__storage_sno_to_key = H5PyDict(h5py_storage_inst=self.__h5py_store, group_inst=self.__group_inst,
                                             h5py_group_path=sno_to_key_dbpath)
        self.__storage_key_to_val = H5PyDict(h5py_storage_inst=self.__h5py_store, group_inst=self.__group_inst,
                                             h5py_group_path=key_to_val_dbpath)

        if (self.__storage_key_to_sno is None) or \
                (self.__storage_sno_to_key is None) or \
                (self.__storage_key_to_val is None):
            raise RuntimeError("random dictionary instance failed to initialize")

        if self.__group_inst is None:
            self.__group_inst = self.__h5py_store["/"]

        super().__init__(seq, inst_key_to_sno=self.__storage_key_to_sno,
                         inst_sno_to_key=self.__storage_sno_to_key,
                         inst_key_to_val=self.__storage_key_to_val, **kwargs)

    def create_memory_friendly_copy(self, working_group=TEMP_WORKING_GROUP):
        return H5PyRandDict(group_inst=self.__group_inst, h5py_group_path=working_group)

    @staticmethod
    def prepare_key(key):
        if isinstance(key, int):
            return str(key)

    def __getitem__(self, key):
        key = H5PyRandDict.prepare_key(key)
        return numpy.array(super()[key])

    def __setitem__(self, key, value):

        key = H5PyRandDict.prepare_key(key)

        self.__storage_sno_to_key.set_create_working_mode()
        self.__storage_key_to_sno.set_create_working_mode()
        self.__storage_key_to_val.set_create_replace_working_mode()

        super().__setitem__(key, value)

    def __delitem__(self, key):
        key = H5PyRandDict.prepare_key(key)
        self.__storage_key_to_sno.set_replace_working_mode()
        self.__storage_sno_to_key.set_replace_working_mode()
        self.__storage_key_to_val.set_replace_working_mode()

        super().__delitem__(key)

    def clear(self):
        if self.__h5py_store is not None:
            self.__h5py_store.close()

    def __del__(self):
        self.clear()
