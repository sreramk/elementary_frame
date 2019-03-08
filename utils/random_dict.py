# Copyright (c) 2019 K Sreram, All rights reserved

import random

from utils.exceptions import KeyDoesNotExist


class RandomDict:
    def __init__(self, dict_value=None):

        self.__key_to_sno = {}
        self.__sno_to_key = {}
        self.__key_to_val = {}

        if dict_value is not None:
            for k in dict_value:
                self.add_element(k, dict_value[k])

    def get_element(self, key):
        return self.__key_to_val[key]

    def set_element(self, key, value):
        if key in self.__key_to_val:
            self.__key_to_val[key] = value
        else:
            raise KeyDoesNotExist

    def add_element(self, key, value):
        """

        :param key:
        :param value: The reference is transferred. A new instance does not get created
        :return:
        """
        if key not in self.__key_to_val:
            self.__sno_to_key[len(self.__sno_to_key)] = key
            self.__key_to_sno[key] = len(self.__key_to_sno)

        self.__key_to_val[key] = value

    def delete_element(self, del_key):

        if del_key not in self.__key_to_val:
            return None

        sno_del_key = self.__key_to_sno[del_key]
        sno_end_key = len(self.__key_to_sno) - 1

        end_key = self.__sno_to_key[sno_end_key]

        self.__sno_to_key[sno_del_key], self.__sno_to_key[sno_end_key] = \
            self.__sno_to_key[sno_end_key], self.__sno_to_key[sno_del_key]

        self.__key_to_sno[end_key] = sno_del_key

        del self.__key_to_sno[del_key]
        del self.__sno_to_key[len(self.__sno_to_key) - 1]
        del self.__key_to_val[del_key]

    def random_key(self):
        if len(self.__sno_to_key) == 0:
            raise KeyDoesNotExist
        return self.__sno_to_key[random.randrange(0, len(self.__sno_to_key))]

    def random_value(self):
        return self.__key_to_val[self.random_key()]

    def random_key_value(self):
        key = self.random_key()
        val = self.__key_to_val[key]
        return key, val

    def create_memory_friendly_copy(self):
        """
        This ensures that the new copy has the same value reference objects in them. This is because, value objects in
        the variable self.__key_to_val is usually supposed be large in size. So creating a copy of this class which
        copies the reference to the 'value' fields may be required if multiple book-keeping is required for the same set
        of key-value pairs
        :return:
        """
        return RandomDict(self.__key_to_val)

    def __str__(self):
        return "Key to serial number: " + str(self.__key_to_sno) + \
               "\n" + "Serial number to key: " + str(self.__sno_to_key) + "\n Key to value: " + \
               str(self.__key_to_val)

    def __len__(self):
        return len(self.__key_to_val)

    def __contains__(self, item):
        return item in self.__key_to_val


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
        for k,v in d.items():
            print(k, d[k])

    for k, v in y.items():
        print(y[k])

    x = {"a": A(10), "b": A(20), "c": A(30)}
    y = {"a": A(1), "b": A(2), "c": A(3)}

    y["a"] = x["a"]
    y["b"]= x["b"]
    del x["b"]
    x["a"].set_a(100)
    print("##################")
    display(x)
    print("#################")
    display(y)

