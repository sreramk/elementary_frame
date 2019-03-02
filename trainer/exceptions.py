# copyright (c) 2019 K Sreram, All rights reserved
class TrainerException(Exception):
    """
    Base class to all trainer-related exceptions. This is usually raised if it is known that the exception belongs to
    trainer but it is unknown which exact exception it actually is.
    """
    pass


class ModelBaseExceptions(TrainerException):
    """
    Class of exceptions raised when the appropriate methods aren't overridden.
    """
    pass


class RunTrainerMustBeOverridden(ModelBaseExceptions):
    """
    run_trainer method isn't overridden.
    """
    pass


class RunTestMustBeOverridden(ModelBaseExceptions):
    """
    run_test method isn't overridden.
    """
    pass


class ExecuteModelMustBeOverridden(ModelBaseExceptions):
    """
    Execute model isn't overridden.
    """
