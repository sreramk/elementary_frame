# copyright (c) 2019 K Sreram, All rights reserved.
import numpy


def ensure_numpy_array(img):
    # ensures that the data is always numpy. Storing in numpy is more efficient and less memory-consuming than storing
    # this in an ordinary list, if the size of the data is high.
    if type(img) is not numpy.ndarray:
        return numpy.array(img)
    else:
        return img


def check_if_numpy_array(img):
    return type(img) is numpy.ndarray
