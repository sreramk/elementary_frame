import numpy
import os
import random

import cv2

from typing import List
import collections

from utils import RandomDict


class ImageDSManage:
    """
    This manages the data-set read/write.
    """

    def __init__(self, dir_list, accepted_ext=[".jpeg", ".jpg", ".png", ".bmp"], buffer_priority=1.0,
                 image_buffer_limit=500):
        self.__dir_list = dir_list  # type: List[str]

        self.__image_dir_list = []  # type: List[str]

        self.__accepted_ext = accepted_ext

        self.__image_buffer_limit = image_buffer_limit

        self.__image_buffer_dict = collections.OrderedDict()

        self.__rand_directories = RandomDict()

        self.__buffer_priority = buffer_priority

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

    def get_image_dir_list(self):
        return self.__image_dir_list.copy()

    def get_dir_list(self):
        return self.__dir_list.copy()

    def get_accepted_ext_list(self):
        return self.__accepted_ext.copy()

    def __cache_image(self, directory, img):
        """
        This cache for images ensures that the added image is always added to the very
        end of the dictionary and if the buffer is full, the oldest or the least frequently
        accessed image is removed. This will usually be the first record of the ordered
        dictionary.
        :param directory:
        :param img:
        :return:
        """
        if len(self.__image_buffer_dict) >= self.__image_buffer_limit:
            del_key = next(iter(self.__image_buffer_dict))
            del self.__image_buffer_dict[del_key]
            self.__rand_directories.delete_element(del_key)

        self.__image_buffer_dict[directory] = img
        self.__rand_directories.add_element(directory, img)

    def __retrieve_img_cache(self, directory):
        """
        Performs image-retrieval from `__image_buffer_dict' ensuring that the most recently
        accessed image is least likely to be deleted upon buffer-overflow (which is constrained
        by `__image_buffer_limit`).
        :param directory:
        :return:
        """
        if directory in self.__image_buffer_dict:
            result = self.__image_buffer_dict[directory]
            del self.__image_buffer_dict[directory]

            # update position, to mark the most recent access
            self.__image_buffer_dict[directory] = result
            return result

    @staticmethod
    def __size_embedded_directory(directory, min_x_f=None, min_y_f=None):
        size_directory = None
        if min_y_f is not None:
            size_directory = directory + "(" + str(min_y_f) + "_" + str(min_x_f) + ")"
            if min_x_f is None:
                raise Exception("Error, min_x_f must not be none or, min_y_f must be none. "
                                "either both (min_x_f, min_y_f) min x value "
                                "and min y value must be assigned, or they must both not be"
                                "simultaneously assigned")
        elif min_x_f is not None:
            raise Exception("Error, min_x_f must be none or, min_y_f must not be none. "
                            "either both (min_x_f, min_y_f) min x value "
                            "and min y value must be assigned, or they must both not be"
                            "simultaneously assigned")
        return size_directory if size_directory is not None else directory

    def __load_image(self, directory):

        """
        Performs a cached load, to reduce the number of file accesses. The number of
        file access decrease as the size of `__image_buffer_limit` increases
        :param directory:
        :return:
        """

        result = self.__retrieve_img_cache(directory)

        if result is not None:
            return result

        new_img = cv2.imread(directory)
        new_img = numpy.float32(new_img) / 255.0

        self.__cache_image(directory=directory, img=new_img)

        return new_img

    def change_buffer_limit_size(self, new_limit_size):
        """
        Changing buffer limit size must prompt a reorder of the ordered dictionary.
        When elements are deleted from the dictionary while reordering, the oldest
        accessed record is deleted first.
        :param new_limit_size:
        :return:
        """
        self.__image_buffer_limit = new_limit_size

        while len(self.__image_buffer_dict) > self.__image_buffer_limit:
            del self.__image_buffer_dict[next(iter(self.__image_buffer_dict))]

    @staticmethod
    def __crop_img(image, size_x, size_y, pos_x, pos_y):
        start_x = pos_x
        start_y = pos_y
        end_x = size_x + start_x
        end_y = size_y + start_y
        return image[start_y:end_y, start_x:end_x]

    @staticmethod
    def __random_crop(image, size_x, size_y, pos_x=None, pos_y=None):

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

        return start_x, start_y, image[start_y:end_y, start_x:end_x]

    def __choose_dir(self):

        rand_choose_ds = random.random()
        rand_choose_buf = random.random() * self.__buffer_priority

        if rand_choose_buf > rand_choose_ds and len(self.__rand_directories) > 0:
            rand_dir = self.__rand_directories.random_key()
        else:
            rand_dir = random.choice(self.__image_dir_list)

        return rand_dir

    def get_batch(self, batch_size, down_sample_factor, min_x_f=None, min_y_f=None):

        batch_original = []
        batch_down_sampled = []

        batch_size = min(len(self.__image_dir_list), batch_size)

        min_x = None
        min_y = None

        for i in range(batch_size):

            original_img = self.__load_image(self.__choose_dir())

            # reducing the information contained in the image

            down_sampled = cv2.resize(original_img,
                                      dsize=(int(len(original_img[0]) / down_sample_factor),
                                             int(len(original_img) / down_sample_factor)))

            down_sampled = cv2.resize(down_sampled,
                                      dsize=(len(original_img[0]),
                                             len(original_img)))

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

        return min_x, min_y, batch_down_sampled, batch_original


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
