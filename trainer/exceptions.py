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


class ModelDerivedClassExceptions(TrainerException):
    """
    Common class of exceptions raised within the model-definition methods.
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
    pass


class GetModelMustBeOverridden(ModelBaseExceptions):
    """
    Get model, which must return the tensor operations for computing the result of the model must be overridden
    """
    pass


class GetParameterTensorsMustBeOverridden(ModelBaseExceptions):
    """
    Get parameter tensors, which returns the parameters of the learning model, must be overridden.
    """
    pass


class PrepareTrainTestDatasetMustBeOverridden(ModelBaseExceptions):
    """
    Prepare train test dataset- This must ensure that the training and the testing data-set are initialized and are
    ready to be executed in both performing both the training and the testing process.
    """
    pass


class SetModelSaverInstanceMustBeOverridden(ModelBaseExceptions):
    """
    Set Model Saver Instance- this must be overridden to accept the saver instance for the model.
    """
    pass


class LossUninitialized(ModelDerivedClassExceptions):
    """
    This is raised when the loss function is uninitialized.
    """
    pass


class LossOptimizerUninitialized(ModelDerivedClassExceptions):
    """
    Raised when the adam loss functionality is not initialized.
    """


class RequiredPlaceHolderNotInitialized(ModelDerivedClassExceptions):
    """
    Signifies that one or more placeholders required for the computation is not defined.
    """
    pass


class TrainDatasetNotInitialized(ModelDerivedClassExceptions):
    """
    The training data-set must be initialized.
    """
    pass


class SaveInstanceNotInitialized(ModelDerivedClassExceptions):
    """
    The saver instance has not been initialized
    """


class TestDatasetNotInitialized(ModelDerivedClassExceptions):
    """
    The testing data-set must be initialized.
    """
    pass


class RunTestError(ModelDerivedClassExceptions):
    """
    Raised within the run test method
    """
    pass


class ModelNotInitialized(ModelDerivedClassExceptions):
    """
    Raised when the model is returned when not initialized.
    """
    pass


class ParameterNotInitialized(ModelDerivedClassExceptions):
    """
    Raised when parameter is returned when not initialized
    """
    pass


class InvalidArgumentCombination(ModelDerivedClassExceptions):
    """
    Raised when there are specific rules for passing in values to the argument and if the rules aren't followed. For
    example, if the arguments arg1 and arg2 must not both be set to a non-None value and if both of them are set to
    None, then this exception will most likely be raised.
    """
    pass
