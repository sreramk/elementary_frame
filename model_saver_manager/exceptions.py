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


class InvalidRangeArg(ModelSaverException):
    """
    Raised when the range given to a function is invalid. A few functions accept a range as an input, which is a tuple
    with the first value representing the low_range and the second value representing the high_range. This requires the
    low_range to be lesser than the high_range. This exception is raised when this rule isn't enforced.
    """
    pass


class NoneValueException(ModelSaverException):
    """
    Raised to indicate that a particular value is None.
    """
    pass


class UnimplementedFeature(ModelSaverException):
    """
    Raised when a feature is potentially implementable but still remains un-implemented.
    """
    pass


class MethodMustBeOverridden(ModelSaverException):
    """
    raised when the method isn't overloaded.
    """
    pass


class TFSessionVariableCannotBeNone(ModelSaverException):
    """
    raised when the session argument is None, or not given in the previous calls.
    """
    pass


class UnknownOrUnspecifiedModel(ModelSaverException):
    """
    Raised when the library to use isn't specified. Not all libraries are supported, but it is possible to add support
    for the unsupported libraries.
    """
    pass


class AccessToUninitializedObject(ModelSaverException):
    """
    raised when attempted to use an uninitialized object
    """
    pass


class CheckpointCannotBeFirstRun(ModelSaverException):
    """
    The first run of this checkpoint must not be run here
    """
    pass