# copyright (c) 2019 K Sreram, All rights reserved.

class ModelBase:

    def __init__(self, model_saver_inst):
        self.__model_saver = model_saver_inst

    def prepare_training_dataset(self):
        pass

    def run_train(self, num_of_epochs, **args):
        pass

    def run_test(self, training_set_frac=1, **args):
        pass

    def execute_model(self, model_input=None, **args):
        pass

    def get_model(self, **args):
        pass

    def get_parameter_tensors(self, **args):
        pass
