# copyright (c) 2019 K Sreram, All rights reserved.
from prepare_dataset.sr_image_ds_manager import ImageDSManage
from utils import check_size


class SplitData:
    """

    """

    def __init__(self, dir_list, accepted_ext=None, buffer_priority=1.0,
                 image_buffer_limit=50, training_split_ratio=0.8, data_set_size=1000):

        accepted_ext = SplitData.__init_accepted_ext(accepted_ext)

        self.__image_ds_manage = ImageDSManage(dir_list, accepted_ext,
                                               buffer_priority, image_buffer_limit)

        self.__training_split_ratio = training_split_ratio

        self.__data_set_size = data_set_size

        self.__reset_dataset()

        self.reset_test_iter()
        self.reset_training_iter()

    @staticmethod
    def __init_accepted_ext(accepted_ext):
        if accepted_ext is None:
            return [".jpeg", ".jpg", ".png", ".bmp"]
        else: return accepted_ext

    def __reset_dataset(self):

        self.__training_dataset = None

        self.__testing_dataset = None

        self.__all_dataset = []

    def acquire_and_hold_dataset(self, batch_size, down_sample_factor, min_x_f=None, min_y_f=None):
        """
        This method prepares the training and the testing data-set. Multiple
        calls to this function causes the existing data-set to be overwritten.
        :return:
        """

        self.__reset_dataset()

        for i in range(self.__data_set_size):
            self.__all_dataset.append(self.__image_ds_manage.get_batch(batch_size, down_sample_factor, min_x_f, min_y_f))
            # print(check_size.getsize(self.__all_dataset))

        train_size = int(self.__data_set_size * self.__training_split_ratio)

        self.__training_dataset = self.__all_dataset[:train_size]
        self.__testing_dataset = self.__all_dataset[train_size:self.__data_set_size]
        self.__training_dataset_iter_point = 0
        self.__testing_dataset_iter_point = 0


    def check_train_data_iter_validity(self):
        if self.__training_dataset_iter_point < len(self.__training_dataset):
            return True
        return False

    def check_test_data_iter_validity(self):
        if self.__testing_dataset_iter_point < len(self.__testing_dataset):
            return True
        return False

    def get_next_training_data(self):
        if self.check_train_data_iter_validity():
            result = self.__training_dataset[self.__training_dataset_iter_point]
            self.__training_dataset_iter_point += 1
            return result
        return None

    def get_next_test_data(self):
        if self.check_test_data_iter_validity():
            result = self.__testing_dataset[self.__testing_dataset_iter_point]
            self.__testing_dataset_iter_point += 1
            return result
        return None

    def get_train_data(self):
        if self.check_train_data_iter_validity():
            return self.__training_dataset[self.__training_dataset_iter_point]
        return None

    def get_test_data(self):
        if self.check_test_data_iter_validity():
            return self.__testing_dataset[self.__testing_dataset_iter_point]
        return None

    def reset_training_iter(self):
        self.__training_dataset_iter_point = 0

    def reset_test_iter(self):
        self.__testing_dataset_iter_point = 0