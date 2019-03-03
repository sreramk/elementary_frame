# copyright (c) 2019 K Sreram, All rights reserved.
import cv2

import numpy
import matplotlib.pyplot as plt

from trainer.exceptions import RunTrainerMustBeOverridden, RunTestMustBeOverridden, ExecuteModelMustBeOverridden, \
    GetModelMustBeOverridden, GetParameterTensorsMustBeOverridden, PrepareTrainTestDatasetMustBeOverridden, \
    SetModelSaverInstanceMustBeOverridden


class ModelBase:

    def __init__(self):
        pass

    @staticmethod
    def display_image(img, black_and_white=False):

        if black_and_white:
            temp = []
            for i in range(len(img)):
                temp.append([])
                for j in range(len(img[0])):
                    temp[i].append([img[i][j], img[i][j], img[i][j]])
            img = numpy.array(temp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(img, cmap='gray')
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(img)
        plt.colorbar()
        plt.grid(False)
        plt.show()

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
