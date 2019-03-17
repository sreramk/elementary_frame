# copyright (c) 2019 K Sreram, All rights reserved
import collections

import h5py

from utils.exceptions import InitializationError


class H5PyDict(collections.MutableMapping):
    WORKING_MODE_CREATE = "create"
    WORKING_MODE_REPLACE = "replace"
    WORKING_MODE_CREATE_REPLACE = "create_replace"  # causes overhead of first verifying if the field exists, while

    # writing.

    @staticmethod
    def prepare_path(cur_path):
        if cur_path[len(cur_path) - 1] != "/":
            return cur_path + "/"
        return cur_path

    def __init__(self, h5py_storage_inst=None, group_inst=None, working_dir=None, database_name=None,
                 h5py_group_path=None, seq=None, **kwargs):

        # this initialization makes a series of assumptions. This prioritizes not having to instantiate anything, if
        # possible. If all the arguments in the initialization are given, then this would cause a conflict, and thus,
        # this conflict is resolved by the above prioritizing rule. The argument list of this init method, loosely
        # describes the order in which the information is prioritized. For example, when h5py_storage_inst is given,
        # and even if all the other arguments are given, the initialization based on file-path information does not
        # happen. That is, the path to the directory and the database name gets completely ignored.
        #
        # But the 'working_group' parameter, which is a string pointing to the path of the storage within the database,
        # will be used if 'group_inst' is not set.

        self.__h5py_storage = None
        self.__storage = None
        self.__working_mode = H5PyDict.WORKING_MODE_CREATE_REPLACE

        if h5py_storage_inst is not None:
            self.__h5py_storage = h5py_storage_inst
            self.__storage = h5py_storage_inst["/"]
        elif group_inst is not None:
            self.__storage = group_inst
        elif h5py_storage_inst is None:
            self.__database_name = database_name
            self.__working_dir = working_dir

            # main file system path. Not the database file system.
            self.__storage_path = H5PyDict.prepare_path(self.__working_dir) + database_name + ".h5"

            self.__h5py_storage = h5py.File(self.__storage_path, 'a')
            self.__storage = self.__h5py_storage["/"]

        if h5py_group_path is not None:
            self.__storage = self.__storage.require_group(h5py_group_path)

        if self.__storage is None:
            raise InitializationError(" HDF5 database storage instance failed to initialize")

        self.update(seq, **kwargs)

    def set_create_working_mode(self):
        self.__working_mode = H5PyDict.WORKING_MODE_CREATE

    def set_replace_working_mode(self):
        self.__working_mode = H5PyDict.WORKING_MODE_REPLACE

    def set_create_replace_working_mode(self):
        self.__working_mode = H5PyDict.WORKING_MODE_CREATE_REPLACE

    def is_mode_create(self):
        return self.__working_mode is H5PyDict.WORKING_MODE_CREATE

    def is_mode_replace(self):
        return self.__working_mode is H5PyDict.WORKING_MODE_REPLACE

    def is_mode_create_replace(self):
        return self.__working_mode is H5PyDict.WORKING_MODE_CREATE_REPLACE

    def __create_or_replace(self, key, value):
        force_create = False
        force_replace = False

        if self.is_mode_create_replace():
            if key not in self.__storage:
                force_create = True
            else:
                force_replace = True

        if self.is_mode_create() or force_create:
            self.__storage[key] = value
        elif self.is_mode_replace() or force_replace:
            del self.__storage[key]
            self.__storage[key] = value

    def update(self, seq, **kwargs):

        if kwargs is not None:
            for k in kwargs:
                self.__create_or_replace(k, kwargs[k])

        if isinstance(seq, collections.MutableMapping):
            for k in seq:
                self.__create_or_replace(k, seq[k])

    def get_h5py_storage_inst(self):
        return self.__h5py_storage

    def get_storage_inst(self):
        return self.__storage

    def clear(self):
        self.__storage.clear()

    def setdefault(self, key, default=0):
        self.set_create_replace_working_mode()
        self.__create_or_replace(key, default)

    def get(self, key):
        return self.__getitem__(key)

    def items(self):
        return self.__storage.items()

    def keys(self):
        return self.__storage.keys()

    def values(self):
        return self.__storage.values()

    def __contains__(self, o):
        return self.__storage.__contains__(o)

    def __setitem__(self, k, v):
        self.__create_or_replace(k, v)

    def __delitem__(self, key):
        del self.__storage[key]

    def __getitem__(self, key):
        return self.__storage[key]

    def __len__(self):
        return len(self.__storage)

    def __iter__(self):
        return iter(self.__storage)

    def __del__(self):
        self.close()

    def close(self):
        if self.__h5py_storage is not None:
            self.__h5py_storage.close()
