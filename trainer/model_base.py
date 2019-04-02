# copyright (c) 2019 K Sreram, All rights reserved.

from abc import ABC, abstractmethod


class ModelBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def set_model_saver_inst(self, **args):
        """
        This must be overridden to accept a model saver instance.
        :param args:
        :return:
        """
        pass

    @abstractmethod
    def prepare_train_test_dataset(self, **args):
        pass

    @abstractmethod
    def run_train(self, num_of_epochs, **args):
        pass

    @abstractmethod
    def run_test(self, **args):
        pass

    @abstractmethod
    def execute_model(self, model_input=None, **args):
        """
        Accepts an input to compute an output to the model. If it is false, then one of the test data must be
        executed.
        :param model_input:
        :param args:
        :return:
        """
        pass

    @abstractmethod
    def get_model(self, **args):
        """
        This must return the model that performs the computation
        :param args:
        :return:
        """
        pass

    @abstractmethod
    def get_parameter_tensors(self, **args):
        """
        Must be overridden to return the parameters of the model.
        :param args:
        :return:
        """
        pass
