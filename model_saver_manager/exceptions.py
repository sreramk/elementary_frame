class ModelSaverException(Exception):
    """
    Base class for all model_saver exceptions. When this is thrown, it can be assumed that model_saver had thrown the
    exception.
    """


class ArgumentMustBeAListOfTensors(ModelSaverException):
    """
    This ensures that the argument given to the method: ModelSaver.TensorFlowParamsHanlde.__init__(self, tensors)
    is a list of tensors.
    """
    pass


class InvalidArgumentType(ModelSaverException):
    """
    This exception is thrown when an invalid type is encountered in the context of model_saver.
    """
    pass


class InvalidArgument(ModelSaverException):
    """
    Raised when the particular argument passed to a method is invalid and execution of which may lead to undefined
    behavior.
    """


class InvalidNumberOfArgumentsPassed(ModelSaverException):
    """
    Certain methods accept only a certain number of arguments. When such a condition is violated, this exception is
    raised.
    """
    pass


class InvalidLeftRightArgumentCombination(ModelSaverException):
    """
    The query method in the ModelSaver class, accepts arguments to determine the comparision brackets. The constant
    CLOSED represents the brackets "[" , "]" and the constant open represents "(", ")". So the combination left_range =
    CLOSED and right_range = OPEN represents the following range: [x,y).
    """
    pass


class MoreThanOneArgumentMustNotBeAssigned(ModelSaverException):
    """This exception prevents functions from having more than one of the arguments assigned
    to a value other than None. This is useful when all the argument gives is an option as to
    which arguments to be passed, but where effectively only one argument must be passed."""
    pass


class InvalidCheckpointID(ModelSaverException):
    """
    Raised when the checkpoint used is invalid. Valid values include all the added checkpoint
    IDs if the IN_GET_METHOD flag is used. Else, the valid range includes anything from 0 to infinity.
    """
    pass


class NoneValueException(ModelSaverException):
    """
    Raised to indicate that a particular value is None.
    """
    pass