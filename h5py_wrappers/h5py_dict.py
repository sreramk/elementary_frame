# copyright (c) 2019 K Sreram, All rights reserved
import collections

import h5py

from utils.exceptions import InitializationError
from h5py_wrappers import H5PyWorkingMode


class H5PyDict(collections.MutableMapping):

    @staticmethod
    def prepare_path(cur_path):
        if cur_path[len(cur_path) - 1] != "/":
            return cur_path + "/"
        return cur_path

    def __init__(self, h5py_group_inst, seq=None, **kwargs):

        self.__storage = h5py_group_inst
        self.__working_mode = H5PyWorkingMode()

        self.__working_mode.set_create_replace_working_mode()

        if not isinstance(self.__storage, h5py.Group) and not isinstance(self.__storage, h5py.File):
            raise InitializationError(" A valid HDF5 database storage instance is needed")

        self.update(seq, **kwargs)

    def get_working_mode(self):
        return self.__working_mode

    def __create_or_replace(self, key, value):
        force_create = False
        force_replace = False

        if self.__working_mode.is_mode_create_replace():
            if key not in self.__storage:
                force_create = True
            else:
                force_replace = True

        if self.__working_mode.is_mode_create() or force_create:
            self.__storage[key] = value
        elif self.__working_mode.is_mode_replace() or force_replace:
            del self.__storage[key]
            self.__storage[key] = value

    def update(self, seq, **kwargs):

        if kwargs is not None:
            for k in kwargs:
                self.__create_or_replace(k, kwargs[k])

        if isinstance(seq, collections.MutableMapping):
            for k in seq:
                self.__create_or_replace(k, seq[k])

    def get_storage_inst(self):
        return self.__storage

    def clear(self):
        self.__storage.clear()

    def setdefault(self, key, default=0):
        self.__working_mode.set_create_replace_working_mode()
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

    def __str__(self):
        return str(dict(self.__storage.items()))

def run():
    hf = h5py.File('/media/sreramk/storage-main/elementary_frame/test_dbs/data.h5', 'a')
    g = hf.require_group("/hello/world/dbstore2/")
    print (g)
    dict_inst = H5PyDict(g)
    dict_inst["a"] = [1,2,3]
    dict_inst["b"] = [1,2,3]
    dict_inst["c"] = [1,2,3]
    print (str(dict_inst))
if __name__ == '__main__':
    run()