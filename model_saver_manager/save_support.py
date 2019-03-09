# copyright (c) 2019 K Sreram, All rights reserved
from model_saver_manager.exceptions import MethodMustBeOverridden


class SaveSupport:
    def __init__(self):
        pass

    def change_tensors(self, tf_variable_list):
        raise MethodMustBeOverridden
        pass

    def get_pyprams(self, **args):
        raise MethodMustBeOverridden
        pass

    def set_tensors(self, pyprams, **args):
        raise MethodMustBeOverridden
        pass
