# Copyright (c) 2019 K Sreram, All rights reserved
import scipy
from scipy import ndimage
import numpy
import os
import random

import cv2

from typing import List
import collections

from utils import RandomDict
from utils import check_size


class ImageDSManage:
    """
    This manages the data-set read/write.
    """

    def __init__(self, dir_list, accepted_ext=[".jpeg", ".jpg", ".png", ".bmp"], buffer_priority=1.0,
                 image_buffer_limit=500, buffer_priority_acceleration=0, buffer_priority_cap=500):
        self.__dir_list = dir_list  # type: List[str]

        self.__image_dir_list = []  # type: List[str]

        self.__accepted_ext = accepted_ext

        self.__image_buffer_limit = image_buffer_limit

        self.__main_img_cache = collections.OrderedDict()

        self.__main_cache_rand = RandomDict()

        self.__down_sampled_img_cache = collections.OrderedDict()

        # self.__down_sampled_img_rand_cache = RandomDict()

        self.__buffer_priority = buffer_priority

        self.__buffer_priority_acceleration = buffer_priority_acceleration

        self.__buffer_priority_cap = buffer_priority_cap

        for s in self.__dir_list:
            temp_dirs = list(os.listdir(s))

            for i in range(len(temp_dirs)):
                temp_dirs[i] = s + temp_dirs[i]

            new_dirs = []

            for strings in temp_dirs:
                for extInst in self.__accepted_ext:
                    if extInst in strings[len(strings) - len(extInst):]:
                        new_dirs.append(strings)
            self.__image_dir_list.extend(list(new_dirs))

    @staticmethod
    def ensure_numpy_array(img):
        if type(img) is not numpy.ndarray:
            return numpy.array(img)
        else:
            return img

    def get_image_dir_list(self):
        return self.__image_dir_list.copy()

    def get_dir_list(self):
        return self.__dir_list.copy()

    def get_accepted_ext_list(self):
        return self.__accepted_ext.copy()

    def __cache_down_sampled_img(self, directory, img):
        """
        A cache for downsampled images which ensures that the added image is always added to
        the very end of the dictionary and if the buffer is full, the oldest or the least
        frequently accessed image is removed. This will usually be the first record of the
        ordered dictionary.
        :param directory:
        :param img:
        :return:
        """

        img = ImageDSManage.ensure_numpy_array(img)

        if len(self.__down_sampled_img_cache) >= self.__image_buffer_limit:
            del_key = next(iter(self.__down_sampled_img_cache))
            del self.__down_sampled_img_cache[del_key]

        self.__down_sampled_img_cache[directory] = img

    def __retrieve_down_sampled_img_cache(self, directory):
        """
        Performs image-retrieval from `__down_sampled_img_cache' ensuring that the most recently
        accessed image is least likely to be deleted upon buffer-overflow (which is constrained
        by `__image_buffer_limit`). This ensures that the first element of the ordered dictionary
        is the most old and the most rarely used record.
        :param directory:
        :return:
        """
        if directory in self.__down_sampled_img_cache:
            result = ImageDSManage.ensure_numpy_array(self.__down_sampled_img_cache[directory])
            del self.__down_sampled_img_cache[directory]

            # update position, to mark the most recent access
            self.__down_sampled_img_cache[directory] = result
            return result

    def __cache_main_image(self, directory, img):
        """
        A cache for images ensures that the added image is always added to the very
        end of the dictionary and if the buffer is full, the oldest or the least frequently
        accessed image is removed. This will usually be the first record of the ordered
        dictionary. This also prepares for random dictionary-key selection.
        :param directory:
        :param img:
        :return:
        """
        img = ImageDSManage.ensure_numpy_array(img)
        if len(self.__main_img_cache) >= self.__image_buffer_limit:
            del_key = next(iter(self.__main_img_cache))
            del self.__main_img_cache[del_key]
            self.__main_cache_rand.delete_element(del_key)

        self.__main_img_cache[directory] = img
        self.__main_cache_rand.add_element(directory, img)

    def __retrieve_main_img_cache(self, directory):
        """
        Performs image-retrieval from `__main_img_cache' ensuring that the most recently
        accessed image is least likely to be deleted upon buffer-overflow (which is constrained
        by `__image_buffer_limit`). This ensures that the first element of the ordered dictionary
        is the most old and the most rarely used record.
        :param directory:
        :return:
        """
        if directory in self.__main_img_cache:
            result = self.__main_img_cache[directory]
            del self.__main_img_cache[directory]

            # update position, to mark the most recent access
            self.__main_img_cache[directory] = ImageDSManage.ensure_numpy_array(result)
            return result

    def __load_or_create_down_sampled_img(self, directory, down_sample_factor):
        """
        Performs a cached load, to reduce the number of file accesses. The number of
        file access decrease as the size of `__image_buffer_limit` increases.
        :param directory:
        :return:
        """

        result = self.__retrieve_down_sampled_img_cache(directory)

        if result is not None:
            return result

        original_img = self.__load_main_image(directory)

        down_sampled = cv2.resize(original_img,
                                  dsize=(int(len(original_img[0]) / down_sample_factor),
                                         int(len(original_img) / down_sample_factor)),
                                  interpolation=cv2.INTER_NEAREST)

        down_sampled = cv2.resize(down_sampled,
                                  dsize=(len(original_img[0]),
                                         len(original_img)),
                                  interpolation=cv2.INTER_NEAREST)

        self.__cache_down_sampled_img(directory, down_sampled)

        return down_sampled

    def __load_main_image(self, directory):

        """
        Performs a cached load, to reduce the number of file accesses. The number of
        file access decrease as the size of `__image_buffer_limit` increases.
        :param directory:
        :return:
        """

        result = self.__retrieve_main_img_cache(directory)

        if result is not None:
            return result

        new_img = cv2.imread(directory)
        new_img = numpy.float32(new_img) / 255.0

        new_img = ImageDSManage.ensure_numpy_array(new_img)

        self.__cache_main_image(directory=directory, img=new_img)

        return new_img

    def change_buffer_limit_size(self, new_limit_size):
        """

        TODO: possible bug. The dictionary iterator changes during iteration. Which will not be allowed.
        Changing buffer limit size must prompt a reorder of the ordered dictionary.
        When elements are deleted from the dictionary while reordering, the oldest
        accessed record is deleted first.
        :param new_limit_size:
        :return:
        """
        self.__image_buffer_limit = new_limit_size

        while len(self.__main_img_cache) > self.__image_buffer_limit:
            del self.__main_img_cache[next(iter(self.__main_img_cache))]

    @staticmethod
    def __crop_img(image, size_x, size_y, pos_x, pos_y):
        start_x = pos_x
        start_y = pos_y
        end_x = size_x + start_x
        end_y = size_y + start_y
        return ImageDSManage.ensure_numpy_array(image[start_y:end_y, start_x:end_x])

    @staticmethod
    def __random_crop(image, size_x, size_y):

        delta_x = len(image[0]) - size_x
        start_x = 0
        if delta_x != 0:
            start_x = random.randrange(0, delta_x)

        delta_y = len(image) - size_y
        start_y = 0
        if delta_y != 0:
            start_y = random.randrange(0, delta_y)

        end_x = size_x + start_x
        end_y = size_y + start_y

        return start_x, start_y, ImageDSManage.ensure_numpy_array(image[start_y:end_y, start_x:end_x])

    def __choose_dir(self):

        rand_choose_ds = random.random()
        rand_choose_buf = random.random() * self.__buffer_priority

        if rand_choose_buf > rand_choose_ds and len(self.__main_cache_rand) > 0:
            rand_dir = self.__main_cache_rand.random_key()
        else:
            rand_dir = random.choice(self.__image_dir_list)

        return rand_dir

    def acceleration_step(self):
        if self.__buffer_priority_acceleration != 0 and self.__buffer_priority < self.__buffer_priority_cap:
            self.__buffer_priority += self.__buffer_priority_acceleration * self.__buffer_priority

    def get_buffer_priority(self):
        return self.__buffer_priority

    def get_batch(self, batch_size, down_sample_factor, min_x_f=None, min_y_f=None):

        batch_original = []
        batch_down_sampled = []

        batch_size = min(len(self.__image_dir_list), batch_size)

        min_x = None
        min_y = None

        for i in range(batch_size):

            directory = self.__choose_dir()
            original_img = self.__load_main_image(directory)

            # reducing the information contained in the image

            down_sampled = self.__load_or_create_down_sampled_img(directory, down_sample_factor)

            if (min_y is None) or (min_y > len(original_img)):
                min_y = len(original_img)

            if (min_x is None) or (min_x > len(original_img[0])):
                min_x = len(original_img[0])

            batch_original.append(original_img)
            batch_down_sampled.append(down_sampled)

        if min_x_f is not None or min_y_f is not None:
            if min_x_f < min_x:
                min_x = min_x_f
            if min_y_f < min_y:
                min_y = min_y_f

        for i in range(batch_size):
            pos_x, pos_y, batch_down_sampled[i] = ImageDSManage.__random_crop(batch_down_sampled[i], min_x, min_y)
            batch_original[i] = ImageDSManage.__crop_img(batch_original[i], min_x, min_y, pos_x, pos_y)

        return min_x, min_y, ImageDSManage.ensure_numpy_array(batch_down_sampled), \
               ImageDSManage.ensure_numpy_array(batch_original)


if __name__ == "__main__":
    # img = cv2.imread("/home/sreramk/PycharmProjects/neuralwithtensorgpu/dataset/DIV2K_train_HR/0001.png")
    # cropped = img[10:1000, 10:1000]
    # cv2.imshow("name", cropped)
    # cv2.waitKey(0)

    img_manager = ImageDSManage(["/home/sreramk/PycharmProjects/neuralwithtensorgpu/dataset/DIV2K_train_HR/"])
    min_x, min_y, batch_down_sampled, batch_original = img_manager.get_batch(10, 10)
    first_img_org = batch_original[0]
    first_img_down = batch_down_sampled[0]
    print(first_img_down)
    cv2.imshow("original", first_img_org)
    cv2.imshow("down", first_img_down)
    cv2.waitKey(0)
