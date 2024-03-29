# Copyright (c) 2019 K Sreram, All rights reserved

import collections
import random

from utils.exceptions import KeyDoesNotExist


class RandomDict(collections.MutableMapping):
    KEY_TO_SNO = "key_to_sno"
    SNO_TO_KEY = "sno_to_key"
    KEY_TO_VAL = "key_to_val"

    def __init__(self, seq=None, inst_key_to_sno=None,
                 inst_sno_to_key=None, inst_key_to_val=None, **kwargs):

        if inst_key_to_sno is None or inst_sno_to_key is None or inst_key_to_val is None:
            inst_key_to_sno = {}
            inst_sno_to_key = {}
            inst_key_to_val = {}

        self._key_to_sno = inst_key_to_sno
        self._sno_to_key = inst_sno_to_key
        self._key_to_val = inst_key_to_val

        self.update(seq, **kwargs)

    def get_element(self, key):
        return self._key_to_val[key]

    def __set_element_low(self, key, value):
        self._key_to_val[key] = value

    def set_element(self, key, value):
        if key in self._key_to_val:
            self.__set_element_low(key, value)
        else:
            raise KeyDoesNotExist

    def add_element(self, key, value):
        """

        :param key:
        :param value: The reference is transferred. A new instance does not get created
        :return:
        """
        if key not in self._key_to_val:
            self._sno_to_key[len(self._sno_to_key)] = key
            self._key_to_sno[key] = len(self._key_to_sno)

        self.__set_element_low(key, value)

    def delete_element(self, del_key):

        if del_key not in self._key_to_val:
            return None

        sno_del_key = self._key_to_sno[del_key]
        sno_end_key = len(self._key_to_sno) - 1

        end_key = self._sno_to_key[sno_end_key]

        self._sno_to_key[sno_del_key], self._sno_to_key[sno_end_key] = \
            self._sno_to_key[sno_end_key], self._sno_to_key[sno_del_key]

        self._key_to_sno[end_key] = sno_del_key

        del self._key_to_sno[del_key]
        del self._sno_to_key[len(self._sno_to_key) - 1]
        del self._key_to_val[del_key]

    def random_key(self):
        if len(self._sno_to_key) == 0:
            raise KeyDoesNotExist
        return self._sno_to_key[random.randrange(0, len(self._sno_to_key))]

    def random_value(self):
        return self.get_element(self.random_key())

    def random_key_value(self):
        key = self.random_key()
        val = self.get_element(key)
        return key, val

    def create_memory_friendly_copy(self):
        """
        This ensures that the new copy has the same value reference objects in them. This is because, value objects in
        the variable self.__key_to_val is usually supposed be large in size. So creating a copy of this class which
        copies the reference to the 'value' fields may be required if multiple book-keeping is required for the same set
        of key-value pairs
        :return:
        """
        return RandomDict(self._key_to_val)

    def update(self, seq, **kwargs):
        if kwargs is not None:
            for k in kwargs:
                self.add_element(k, kwargs[k])

        if isinstance(seq, collections.MutableMapping):
            for k in seq:
                self.add_element(k, seq[k])

    def __str__(self):
        return "Key to serial number: " + str(self._key_to_sno) + \
               "\n" + "Serial number to key: " + str(self._sno_to_key) + "\n Key to value: " + \
               str(self._key_to_val)

    def __len__(self):
        return len(self._key_to_val)

    def __contains__(self, item):
        return item in self._key_to_val

    def __delitem__(self, key):
        self.delete_element(key)

    def __getitem__(self, key):
        return self.get_element(key)

    def __iter__(self):
        return iter(self._key_to_val)

    def __setitem__(self, key, value):
        self.add_element(key, value)

    def items(self):
        return self._key_to_val.items()

    def clear(self):
        self._key_to_sno.clear()
        self._sno_to_key.clear()
        self._key_to_val.clear()


if __name__ == "__main__":
    x = RandomDict({1: 2, -4: 5, -7: 11, 4: 3, 100: 200, 9: 9})

    x.delete_element(1)
    x.delete_element(-7)
    x.delete_element(9)
    x.delete_element(4)

    print(str(x))
    print(x.random_value())
    print(x.random_value())
    print(x.random_value())
    print(x.random_value())
    print(x.random_value())
    print(x.random_value())
    print(x.random_value())
    print(x.random_value())

    print("############################")


    class A:
        def __init__(self, v):
            self.__a = v

        @staticmethod
        def modify(b):
            b.__a = 200

        def get_a(self):
            return self.__a

        def set_a(self, v):
            self.__a = v

        def __str__(self):
            return str(self.get_a())


    x = A(100)
    y = A(100)
    print(x.get_a())
    print(y.get_a())

    x.modify(y)
    print(y.get_a())

    x = {"a": A(1), "b": A(2), "c": A(3)}
    print(x["a"])
    y = {}
    for k, v in x.items():
        y[k] = v
    x['c'].set_a(10)
    print("#####################")


    def display(d):
        for k, v in d.items():
            print(k, d[k])


    for k, v in y.items():
        print(y[k])

    x = {"a": A(10), "b": A(20), "c": A(30)}
    y = {"a": A(1), "b": A(2), "c": A(3)}

    y["a"] = x["a"]
    y["b"] = x["b"]
    del x["b"]
    x["a"].set_a(100)
    print("##################")
    display(x)
    print("#################")
    display(y)
