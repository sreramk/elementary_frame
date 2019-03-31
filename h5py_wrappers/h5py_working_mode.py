# copyright (c) 2019 K Sreram, All rights reserved.

class H5PyWorkingMode:
    WORKING_MODE_CREATE = "create"
    WORKING_MODE_REPLACE = "replace"
    WORKING_MODE_CREATE_REPLACE = "create_replace"  # causes overhead of first verifying if the field exists, while
    # writing.

    def __init__(self):
        self.__working_mode = None
        self.set_create_replace_working_mode()

    def set_create_working_mode(self):
        self.__working_mode = H5PyWorkingMode.WORKING_MODE_CREATE

    def set_replace_working_mode(self):
        self.__working_mode = H5PyWorkingMode.WORKING_MODE_REPLACE

    def set_create_replace_working_mode(self):
        self.__working_mode = H5PyWorkingMode.WORKING_MODE_CREATE_REPLACE

    def is_mode_create(self):
        return self.__working_mode is H5PyWorkingMode.WORKING_MODE_CREATE

    def is_mode_replace(self):
        return self.__working_mode is H5PyWorkingMode.WORKING_MODE_REPLACE

    def is_mode_create_replace(self):
        return self.__working_mode is H5PyWorkingMode.WORKING_MODE_CREATE_REPLACE
