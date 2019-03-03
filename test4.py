import sys

import numpy

from utils.check_size import get_size

input()
def fnc(flag=True):
    while True:
        v = [] # numpy.zeros((10000, 10000))
        print("Hit")
        for i in range(10000):
            v.append([])
            for j in range(10000):
                v[i].append(i + j)
        v = numpy.array(v)
        if flag == False:
            input()
            break
    return v

fnc(False)