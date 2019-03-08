# copyright (c) 2019 K Sreram, All rights reserved


class PrepareDataSetException(Exception):
    """
    Exceptions belonging to the prepare-data set modules
    """
    pass


class DataSetManageException(PrepareDataSetException):
    """
    Exceptions raised within the ImageDataSetManager class
    """
    pass


class DataBufferException(PrepareDataSetException):
    """
    Raised in the BufferImageStorage class
    """
    pass


class BufferOverflow(DataBufferException):
    """
    Raised when the total number of data-points attempted to be added exceeds the assigned limit.
    """
    pass


class BufferIsEmpty(DataBufferException):
    """
    Raised when attempted to get a random value while the buffer is empty
    """
    pass


class InvalidFlag(DataBufferException):
    """
    Raised when the flag given is invalid and cannot be handled.
    """
    pass


class ImageNotFound(DataBufferException):
    """
    Raised when the image is not found in the buffer.
    """
    pass


class UniqueGetIsNotInitialized(DataBufferException):
    """
    raised when get_random_image_unique method is called without initializing unique get.
    """
    pass



class InvalidDataSetLabel(DataSetManageException):
    """
    The ds_label or "data set label" can either be "training" or "testing". If it is anything else, this exception is
    raised.
    """
    pass


class MethodNotOverridden(DataSetManageException):
    """
    Raised when a method that have to be overridden was not overridden.
    """
    pass


class InvalidDataPointDefinition(DataSetManageException):
    """
    A data-point must contain one input and one expected output field.
    """
    pass


class StopPopulatingTrainOrTestBuffer(DataSetManageException):
    """
    This is raised when the overridden _get_data method no longer has any data to return.
    """
    pass