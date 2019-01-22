# Copyright (c) 2019 K Sreram, All rights reserved

import random


class RandomDict:
    def __init__(self, dict_value = {}):

        self.__key_to_sno = {}
        self.__sno_to_key = {}
        self.__key_to_val = {}

        for k in dict_value:
            self.add_element(k, dict_value[k])

    def add_element(self, key, value):

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
            return 0
        return self.__sno_to_key[random.randrange(0, len(self.__sno_to_key))]

    def random_value(self):
        return self.__key_to_val[self.random_key()]

    def __str__(self):
        return "Key to serial number: " + str(self.__key_to_sno) + \
               "\n" + "Serial number to key: " + str(self.__sno_to_key) + "\n Key to value: " + \
               str(self.__key_to_val)

    def __len__(self):
        return len(self.__key_to_val)

if __name__ == "__main__":

    x = RandomDict({1:2, -4:5, -7:11, 4:3, 100:200, 9:9})

    x.delete_element(1)
    x.delete_element(-7)
    x.delete_element(9)
    x.delete_element(4)

    print (str(x))
    print (x.random_value())
    print (x.random_value())
    print (x.random_value())
    print (x.random_value())
    print (x.random_value())
    print (x.random_value())
    print (x.random_value())
    print (x.random_value())
