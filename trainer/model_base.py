# copyright (c) 2019 K Sreram, All rights reserved.

from trainer.exceptions import RunTrainerMustBeOverridden, RunTestMustBeOverridden, ExecuteModelMustBeOverridden, \
    GetModelMustBeOverridden, GetParameterTensorsMustBeOverridden, PrepareTrainTestDatasetMustBeOverridden, \
    SetModelSaverInstanceMustBeOverridden


class ModelBase:

    def __init__(self):
        pass

    def set_model_saver_inst(self, **args):
        """
        This must be overridden to accept a model saver instance.
        :param args:
        :return:
        """
        raise SetModelSaverInstanceMustBeOverridden

    def prepare_train_test_dataset(self, **args):
        raise PrepareTrainTestDatasetMustBeOverridden

    def run_train(self, num_of_epochs, **args):
        raise RunTrainerMustBeOverridden

    def run_test(self, **args):
        raise RunTestMustBeOverridden

    def execute_model(self, model_input=None, **args):
        """
        Accepts an input to compute an output to the model. If it is false, then one of the test data must be
        executed.
        :param model_input:
        :param args:
        :return:
        """
        raise ExecuteModelMustBeOverridden

    def get_model(self, **args):
        """
        This must return the model that performs the computation
        :param args:
        :return:
        """
        raise GetModelMustBeOverridden

    def get_parameter_tensors(self, **args):
        """
        Must be overridden to return the parameters of the model.
        :param args:
        :return:
        """
        raise GetParameterTensorsMustBeOverridden
