import sys

from utils.check_size import get_size

input()
def fnc(flag=True):
    while True:
        v = []
        for i in range(10000):
            v.append([])
            for j in range(2000):
                v.append((i, j))
        print(get_size(v))
        if flag == False:
            input()
            break


fnc(False)