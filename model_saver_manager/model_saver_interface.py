# copyright (c) 2019 K Sreram, All rights reserved
from model_saver_manager.model_saver import ModelSaver


class ModelSaverInterface:

    def __init__(self, saver_inst:ModelSaver):
        self.__saver_inst = saver_inst

    def change_working_tensor_prams (self, new_tensor_prams):
        self.__saver_inst.change_working_tensor_prams(new_tensor_prams)

    def checkpoint_model(self):
        pass

    def commit_all (self):
        pass

    def create_check_point(self):
        pass

    def delete_all_saved_states(self):
        pass

    def delete_checkpoint(self):
        pass

    def get_checkpoint_name(self):
        pass

    def load_checkpoint(self):
        pass

    def query_checkpoint_info(self):
        pass

    def query_get_first_record(self):
        pass

    def query_get_last_record(self):
        pass

    def set_first_run(self):
        pass

    def set_not_first_run(self):
        pass
