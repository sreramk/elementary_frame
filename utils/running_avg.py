# copyright (c) 2019 K Sreram, All rights reserved.


class RunningAvg:

    def __init__(self):
        self.__avg = None
        self.__count = None
        self.reset()

    def get_avg(self):
        return self.__avg

    def reset(self):
        self.__avg = 0.0
        self.__count = 0

    def __set_avg(self, avg):
        self.__avg = avg

    def __get_count(self):
        return self.__count

    def __set_count(self, count):
        self.__count = count

    def __increment_count(self):
        self.__count += 1

    def add_to_avg(self, val):

        temp = self.get_avg() * self.__get_count()
        self.__increment_count()
        temp += val
        temp = temp/self.__get_count()
        self.__set_avg(temp)

        return self.get_avg()


