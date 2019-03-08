# copyright (c) 2019 K Sreram, All rights reserved


class UtilsExceptions(Exception):
    """
    Raised in utils functions
    """
    pass


class KeyDoesNotExist(UtilsExceptions):
    """
    raised when the key is not present, while setting the value.
    """
    pass