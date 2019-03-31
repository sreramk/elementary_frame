# copyright (c) 2019 K Sreram, All rights reserved
import numpy

from utils.general_utils import ensure_numpy_array


class GeneralArrayList:

    @staticmethod
    def resize_numpy(store, shape, dtype):
        end_size = len(store)
        temp = store
        store = numpy.empty(shape, dtype=dtype)
        store[0:end_size] = temp[0:end_size]

    def __resize_store(self):
        temp_store_shape = [self.__capacity]
        temp_store_shape.extend(self.__shape)
        if self.__store is None:
            self.__store = self.__data_constructor(temp_store_shape, dtype=self.__dtype)
        else:
            self.__resize_method(self.__store, temp_store_shape, self.__dtype)

    def __init__(self, shape=None, dtype=None, increment_fac=1.3, initial_len=4, data_constructor=numpy.empty,
                 resize_method=resize_numpy):
        self.__data_constructor = data_constructor
        self.__shape = None
        self.__dtype = dtype
        self.__capacity = initial_len
        self.__increment_frac = increment_fac
        self.__end = 0
        self.__store = None
        self.__resize_method = resize_method

        if shape is not None:
            self.__shape = shape
            self.__resize_store()

    def append(self, val: numpy.ndarray):
        val = ensure_numpy_array(val)
        if self.__store is None:
            self.__shape = list(val.shape)
            self.__resize_store()
        elif self.__end >= self.__capacity:
            self.__capacity = int(self.__capacity * self.__increment_frac)
            self.__resize_store()

        self.__store[self.__end] = val
        self.__end += 1

    def make_compact(self):
        self.__capacity = self.__end
        self.__resize_store()

    def pop(self):
        if self.__end <= 0:
            raise IndexError("List size is zero, cannot pop an element")
        self.__end -= 1

    def __getitem__(self, item):
        if self.__end < item or item < 0:
            raise IndexError("Invalid index value. The index value must be strictly within range.")
        return self.__store[item]

    def __setitem__(self, key, value):
        if key < 0 or self.__end < key:
            raise IndexError("Invalid index value. The index value must be strictly within range. "
                             "This cannot be used for inserting new elements. This must be used only to modify "
                             "Old elements.")
        self.__store[key] = value

    def __str__(self):
        return str(self.__store[0:self.__end])

    def __len__(self):
        return self.__end

    def capacity(self):
        return self.__capacity


def run():
    x = GeneralArrayList()
    for i in range(10000):
        x.append(numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0 + float(i)]]))

    print(x)
    x.make_compact()
    print(x.capacity())


if __name__ == '__main__':
    run()
