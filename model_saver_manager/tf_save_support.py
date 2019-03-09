# copyright (c) 2019 K Sreram, All rights reserved.

import numpy
import tensorflow as tf

from model_saver_manager.exceptions import ArgumentMustBeAListOfTensors, TFSessionVariableCannotBeNone
from model_saver_manager.save_support import SaveSupport


class TensorFlowParamsHandle(SaveSupport):
    PREFIX_INP = "placeholder_"

    def __init__(self, tfvariable_list, name):

        super().__init__()

        self.__tf_variable_list_raw = None
        self.__tfvar_inputs = None
        self.__assign_to_tensor_oper = None
        self.change_tensors(tfvariable_list)
        self.__name = name
        self.__make_tensors_ready()

    def __make_tensors_ready(self):
        PHandle = TensorFlowParamsHandle
        self.__tfvar_inputs = []  # placeholder lists
        self.__assign_to_tensor_oper = []  # a temporary storage for the parameter values before being extracted

        for tfvars in self.__tf_variable_list_raw:
            cur_inp = tf.placeholder(dtype=tfvars.dtype, shape=tfvars.get_shape(),
                                     name=PHandle.PREFIX_INP + self.__name)
            self.__tfvar_inputs.append(cur_inp)
            self.__assign_to_tensor_oper.append(tf.assign(tfvars, cur_inp))

    def change_tensors(self, tf_variable_list):
        self.__tf_variable_list_raw = tf_variable_list

        if not isinstance(self.__tf_variable_list_raw, list):
            raise ArgumentMustBeAListOfTensors

        if not all(isinstance(tfvariable, tf.Variable) for tfvariable in self.__tf_variable_list_raw):
            raise ArgumentMustBeAListOfTensors

    def get_pyprams(self, sess):
        result = sess.run(fetches=self.__tf_variable_list_raw)
        for i in range(len(result)):
            result[i] = result[i].tolist()
        return result

    def set_tensors(self, pyprams, sess=None):
        if sess is None:
            raise TFSessionVariableCannotBeNone
        for i in range(len(pyprams)):
            pyprams[i] = numpy.array(pyprams[i])

        def create_feed_dict():
            result = {}
            for i in range(len(self.__tfvar_inputs)):
                result[self.__tfvar_inputs[i]] = pyprams[i]
            return result

        return sess.run(fetches=self.__assign_to_tensor_oper, feed_dict=create_feed_dict())
