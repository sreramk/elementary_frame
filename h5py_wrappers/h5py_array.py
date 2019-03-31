# copyright (c) 2019 K Sreram, All rights reserved.

import h5py
import numpy

from utils.general_array_list import GeneralArrayList


class H5PyArray:
    def __init__(self, name, h5py_group_inst: h5py.Group, shape):
        self.__storage = None
        self.__h5py_group_inst = h5py_group_inst
        self.__name = name
        self.__shape = shape
        self.__dtype = None
        if name not in h5py_group_inst:
            if shape is not None:
                self.__configure_shape(shape)
        else:
            self.__storage = h5py_group_inst[name]

    def __configure_shape(self, shape: list, dtype=None):
        min_shape = [1]
        min_shape.extend(shape)
        max_shape = [None]
        max_shape.extend(shape)
        self.__storage = self.__h5py_group_inst.create_dataset(self.__name, shape=min_shape, maxshape=max_shape,
                                                               dtype=dtype)
        self.__shape = shape
        self.__dtype = dtype

    def __getitem__(self, key):
        return self.__storage[key]

    def __setitem__(self, key, value):
        self.__storage[key] = value

    def generate_data_constructor(self):
        def data_constructor(shape, dtype):
            self.resize(shape)
            # self.__configure_shape(shape, dtype)
            return self

        return data_constructor

    def resize(self, new_size, axis=None):
        self.__storage: h5py.Dataset
        self.__storage.resize(new_size, axis)

    @staticmethod
    def generate_resize_method():

        def resize_method(store: h5py.Dataset, shape, dtype):
            store.resize(shape)

        return resize_method

    def clear(self):
        if self.__name in self.__h5py_group_inst:
            del self.__h5py_group_inst[self.__name]
        self.__configure_shape(self.__shape, self.__dtype)

    def __str__(self):
        return str(numpy.array(self.__storage))


def create_h5py_buffer(name, h5py_group_inst: h5py.Group, shape,
                       dtype=None, increment_fac=1.3, initial_len=4):
    arr_inst = H5PyArray(name, h5py_group_inst, shape)
    return GeneralArrayList(shape, dtype, increment_fac, initial_len,
                            data_constructor=arr_inst.generate_data_constructor(),
                            resize_method=H5PyArray.generate_resize_method())


def run():
    hf = h5py.File('/media/sreramk/storage-main/elementary_frame/test_dbs/data.h5', 'a')
    g = hf.require_group("/hello/world/dbstore/")
    arr_inst2 = create_h5py_buffer("test_array_2", g, shape=[1])
    arr_inst = H5PyArray("test_array", g, shape=[1])
    arr_inst.clear()
    arr_inst.resize([2, 1])

    arr_inst[0] = [10]

    print(str(arr_inst))
    print("#############################")
    arr_inst.resize([4, 1])

    arr_inst[3] = [10]

    print(str(arr_inst))
    print("#############################")
    arr_inst.resize([2, 1])

    arr_inst[1] = [10]

    print(str(arr_inst))

    for i in range(10):
        arr_inst2.append([i])
    print ("############ 2 ###################")
    print (str(arr_inst2))
    print (len(arr_inst2))
    print (arr_inst2.capacity())

if __name__ == "__main__":
    run()
