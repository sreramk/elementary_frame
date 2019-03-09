# copyright (c) 2019 K Sreram, All rights reserved.

# TODO break this class into multiple classes, to distinctly separate each functionality

import contextlib
import copy
import datetime
import json
import os
import time
from collections import OrderedDict

from model_saver_manager.exceptions import InvalidCheckpointID, InvalidArgumentType, \
    InvalidNumberOfArgumentsPassed, InvalidLeftRightArgumentCombination, InvalidRangeArg, UnimplementedFeature, \
    UnknownOrUnspecifiedModel
from model_saver_manager.save_support import SaveSupport
from model_saver_manager.tf_save_support import TensorFlowParamsHandle
from utils.running_avg import RunningAvg


class ModelSaver:
    __DATE_HANDLE_FOR_JSON = lambda obj: (
        obj.isoformat()
        if isinstance(obj, (datetime.datetime, datetime.date))
        else None
    )

    DEFAULT_EXTENSION = ".mdl"
    INDEX_FILE_POSTFIX = "_index"

    STATIC_CHECKPOINT = 1  # replaces the same checkpoint for each iteration.
    DYNAMIC_CHECKPOINT = 2  # creates a new checkpoint for each iteration.

    # The following are the checkpoint constants used. These values are negative as the valid range for the checkpoints
    # starts from zero and these negative values can be used to represent special types of checkpoints like the default
    # checkpoint, the last checkpoint, the latest used checkpoint etc.

    NONE_CHECKPOINT = -1  # represents that no valid checkpoint is used. The checkpoint resolver returns None for this.

    LAST_CHECKPOINT = -2  # the most recent record
    FIRST_CHECKPOINT = -3  # the earliest record that has not been deleted.

    DEFAULT_CHECKPOINT = -4  # represents the currently active record.

    LATEST_CHECK_POINT_ID = -5  # represents the most recently used checkpoint.

    IN_GET_METHOD = True
    NOT_IN_GET_METHOD = False

    OPEN = 0
    CLOSED = 1

    # Value fields:

    H_NUM_OF_CHECK_POINT = "h_num_of_check_point"  # number of checkpoints added
    H_FREE_CHECK_POINT_ID = "h_free_check_point_id"  # free checkpoint ID to be used for assigning a new record
    H_LATEST_CHECK_POINT_ID = "h_latest_check_point_id"  # checkpoint ID of the most recently used checkpoint
    H_CHECK_POINT_STORE = "h_check_point_store"  # the list of checkpoints, and its info.

    # Attribute fields for each set of weight values. This is present in the
    # header and each instance of the soted record

    H_CHECK_POINT_ID = "h_check_point_ID"  # this is the header key in the listing
    H_CHECK_POINT_NAME = "h_check_point_name"  # which is a combination of the ID and the name of the checkpoints stored
    H_CHECK_POINT_TIME_STAMP = "h_check_point_time_stamp"  # timestamp of creating the checkpoint
    H_CHECK_POINT_EFFICIENCY = "h_check_point_efficiency"  # efficiency of the parameters recorded

    # CHECKPOINT_STORE this is an attribute only available in the record and not the header. This is not to be confused
    # with h_check_point_store defined as the symbol H_CHECK_POINT_STORE which lists all the stored record's
    # header-details in the header.

    CHECK_POINT_STORE = "check_point_store"

    # Theory: For each checkpoint, the state of the application at that particular point determines the working level of
    # the ML model. But to prevent the model from deviating from the best checkpoint, it is usually effective to either
    # discard the trained model that performs worse on a testing data-set compared to the previous checkpoint or find
    # the checkpoint that had recorded the best performance and keep reverting back to that. To implement this
    # effectively, the next version of this class must have support for storing few of the most recently used
    # checkpoints to eventually reduce the overall disk access.

    # checkpoint reload strategy:

    REBASE_BEST_CHECKPOINT = "rebase_best_checkpoint"  # search for the best checkpoint and reload it

    REBASE_IGNORE_BAD_CHECKPOINT = "rebase_ ignore_bad_checkpoint"  # Revert to the previous checkpoint if the
    # efficiency of the current is lesser

    REBASE_CHECKPOINT_IGNORE = "rebase_checkpoint_ignore"  # signifies that no rebasing happens for each checkpoint
    # and the currently-active checkpoint will remain active.

    DEFAULT_REBASE_BEST_CHECKPOINT_RADIUS = (-5, 5)

    # As of now, only support for TensorFlow is added. In the future, this library will be generalized to support
    # multiple ML platforms.

    MT_TENSORFLOW = "mt_TensorFlow"
    MT_PYTORCH = "mt_PyTorch"

    class TimeIterSkipManager:

        # These constants define the standard for time-checking implemented in the method: checkpoint_model

        DEFAULT_SKIP = 1
        ST_SECONDS_SKIP = 0
        ST_ITER_SKIP = 1

        def __init__(self, skip_type=ST_ITER_SKIP, skip_duration=DEFAULT_SKIP):
            self.__skip_type = None
            self.__initial_time_stamp = None
            self.__time_skip_duration = None

            self.__iter_skip_duration = None
            self.__iter_skip_counter = None

            self.set_skipper(skip_type, skip_duration)

        def __reset_time(self):
            self.__initial_time_stamp = time.mktime(time.localtime())

        def __reset_iter(self):
            self.__iter_skip_counter = 0

        def __increment_iter(self):
            self.__iter_skip_counter += 1

        def __cur_iter_count(self):
            return self.__iter_skip_counter

        def set_skipper(self, skip_type=ST_ITER_SKIP, skip_duration=DEFAULT_SKIP):
            SkipHandle = ModelSaver.TimeIterSkipManager
            self.__skip_type = skip_type
            if skip_type == SkipHandle.ST_SECONDS_SKIP:
                self.__reset_time()
                self.__time_skip_duration = skip_duration
            elif skip_type == SkipHandle.ST_ITER_SKIP:
                self.__reset_iter()
                self.__iter_skip_duration = skip_duration
            else:
                raise InvalidArgumentType

        def check_execute_checkpoint(self):

            SkipHandle = ModelSaver.TimeIterSkipManager

            if self.__skip_type == SkipHandle.ST_SECONDS_SKIP:
                cur_time_stamp = time.mktime(time.localtime())
                time_diff = cur_time_stamp - self.__initial_time_stamp
                if time_diff >= self.__time_skip_duration:
                    self.__reset_time()
                    return True
                return False

            elif self.__skip_type == SkipHandle.ST_ITER_SKIP:
                cur_iter_count = self.__cur_iter_count()

                if cur_iter_count >= self.__iter_skip_duration:
                    self.__reset_iter()
                    return True
                self.__increment_iter()
                return False
            else:
                raise InvalidArgumentType

    def __init__(self, save_name, tensor_prams, save_file_path=os.getcwd(),
                 check_point_digits=5, extension=DEFAULT_EXTENSION, reset=False, model_type=MT_TENSORFLOW):

        self.__save_file_path = save_file_path  # path to store the checkpoints.
        self.__save_name = save_name  # name of the saving instance. used in naming the files.

        self.__check_point_digits = check_point_digits  # total number of digits for the checkpoint numbering
        self.__extension = extension  # storing file extension
        self.__header = None  # holds the storage header
        self.__header_address = None  # holds the address to the header
        self.__first_run = None  # triggers a different behavior for the first iteration of the check-pointing process

        self.__initialize_model(model_type, tensor_prams, save_name)

        self.__initialize_header(reset)

        self.__checkpoint_record = None  # holds the checkpoint current record. After each checkpoint, this gets
        # replaced by the new checkpoint that needs to be added if the checkpoint strategy is dynamic (which is default)

        self.set_first_run()

        self.__time_iter_skip = None

        self.__running_avg_in_checkpoint_model = None

    @staticmethod
    def __first(s):
        return next(iter(s))

    @staticmethod
    def __last(s):
        return next(reversed(s))

    @staticmethod
    def __create_record_header_structure(checkpoint_name, checkpoint_timestamp,
                                         checkpoint_efficiency):
        return OrderedDict({
            ModelSaver.H_CHECK_POINT_NAME: checkpoint_name,
            ModelSaver.H_CHECK_POINT_TIME_STAMP: checkpoint_timestamp,
            ModelSaver.H_CHECK_POINT_EFFICIENCY: checkpoint_efficiency
        })

    @staticmethod
    def __create_record_structure(checkpoint_id, checkpoint_name, checkpoint_timestamp,
                                  checkpoint_efficiency, checkpoint_store):
        return OrderedDict({ModelSaver.H_CHECK_POINT_ID: checkpoint_id,
                            ModelSaver.H_CHECK_POINT_NAME: checkpoint_name,
                            ModelSaver.H_CHECK_POINT_TIME_STAMP: checkpoint_timestamp,
                            ModelSaver.H_CHECK_POINT_EFFICIENCY: checkpoint_efficiency,

                            ModelSaver.CHECK_POINT_STORE: checkpoint_store
                            })

    @staticmethod
    def __create_header_structure(num_of_checkpoints, free_checkpoint_id, latest_checkpoint_id, checkpoint_store):
        return OrderedDict({ModelSaver.H_NUM_OF_CHECK_POINT: num_of_checkpoints,
                            ModelSaver.H_FREE_CHECK_POINT_ID: free_checkpoint_id,
                            ModelSaver.H_LATEST_CHECK_POINT_ID: latest_checkpoint_id,
                            ModelSaver.H_CHECK_POINT_STORE: checkpoint_store})

    def __initialize_model(self, model_type, tensor_prams, save_name):
        if model_type == ModelSaver.MT_TENSORFLOW:
            self.__converter_instance = TensorFlowParamsHandle(tensor_prams, save_name)
        elif model_type == ModelSaver.MT_PYTORCH:
            raise UnimplementedFeature
        elif isinstance(model_type, SaveSupport):
            raise UnknownOrUnspecifiedModel

    def __initialize_header(self, reset):
        if reset:
            self.delete_all_saved_states()
            self.__load_header(True)
        else:
            self.__load_header(False)

    def __header_initialized(self):
        return self.__header is not None

    def __header_not_initialized(self):
        return self.__header is None

    def __check_if_files_are_corrupt(self):
        pass

    def __get_checkpoint_header(self, checkpoint_id):
        checkpoint_store = self.__header[ModelSaver.H_CHECK_POINT_STORE]
        if len(checkpoint_store) >= checkpoint_id:
            return checkpoint_store[checkpoint_id]
        return None

    def __get_new_checkpoint_id(self):
        new_id = self.__header[ModelSaver.H_FREE_CHECK_POINT_ID]
        self.__header[ModelSaver.H_FREE_CHECK_POINT_ID] = new_id + 1
        return new_id

    def __get_last_assigned_id(self):
        return self.__header[ModelSaver.H_FREE_CHECK_POINT_ID]

    def __resolve_checkpoint_id(self, checkpoint_id, in_get_method, iteration_count=0):
        """

        :param checkpoint_id:
        :param in_get_method:
        :param iteration_count:
        :return:
        """
        checkpoint_id_list = None

        if checkpoint_id is None:
            return None

        if not isinstance(checkpoint_id, int):
            checkpoint_id_list = copy.deepcopy(checkpoint_id)
            if len(checkpoint_id_list) <= iteration_count:
                raise InvalidCheckpointID

            checkpoint_id = checkpoint_id[iteration_count]

        #  the storage to all the checkpoint_id associated meta-details of the record
        checkpoint_store = self.__header[ModelSaver.H_CHECK_POINT_STORE]

        if in_get_method:
            if checkpoint_id in checkpoint_store:
                return checkpoint_id
        else:
            if checkpoint_id >= 0:
                return checkpoint_id

        if len(checkpoint_store) == 0 or checkpoint_id is None:
            return None

        if checkpoint_id == ModelSaver.LAST_CHECKPOINT:
            return ModelSaver.__last(checkpoint_store)
        elif checkpoint_id == ModelSaver.FIRST_CHECKPOINT:
            return ModelSaver.__first(checkpoint_store)
        elif checkpoint_id == ModelSaver.DEFAULT_CHECKPOINT:
            if self.__checkpoint_record is not None:
                return self.__checkpoint_record[ModelSaver.H_CHECK_POINT_ID]
        elif checkpoint_id == ModelSaver.LATEST_CHECK_POINT_ID:
            latest_checkpoint = self.__header[ModelSaver.H_LATEST_CHECK_POINT_ID]
            if latest_checkpoint is not None and latest_checkpoint in checkpoint_store:
                return latest_checkpoint
        elif checkpoint_id == ModelSaver.NONE_CHECKPOINT:
            return None

        if checkpoint_id_list is not None and len(checkpoint_id_list) > iteration_count:
            iteration_count += 1
            return self.__resolve_checkpoint_id(checkpoint_id_list, in_get_method, iteration_count)

        raise InvalidCheckpointID

    @staticmethod
    def __change_h_check_point_store_keys_type(header_data, totype):
        temp = header_data[ModelSaver.H_CHECK_POINT_STORE]
        checkpoint_store = OrderedDict()
        for key, value in temp.items():
            checkpoint_store[totype(key)] = value
        modified_header = ModelSaver.__create_header_structure(header_data[ModelSaver.H_NUM_OF_CHECK_POINT],
                                                               header_data[ModelSaver.H_FREE_CHECK_POINT_ID],
                                                               header_data[ModelSaver.H_LATEST_CHECK_POINT_ID],
                                                               checkpoint_store)
        return modified_header

    def __load_header(self, reset):
        """

        :param reset:
        :return:
        """
        if not reset:
            self.__header_address = self.__get_checkpoint_index_addr()
            if os.path.isfile(self.__header_address):
                with open(self.__header_address, 'r') as input_file:
                    header_json = json.load(input_file, object_pairs_hook=OrderedDict)
                    self.__header = self.__change_h_check_point_store_keys_type(header_json, int)
                return

        self.__new_header()

    def ___save_header(self):
        with open(self.__header_address, 'w') as output_file:
            #  dmp = json.dumps(self.__header, default=ModelSaver.DATE_HANDLE_FOR_JSON)
            safe_header = self.__change_h_check_point_store_keys_type(self.__header, str)
            json.dump(safe_header, output_file, default=ModelSaver.__DATE_HANDLE_FOR_JSON)

    def __delete_header_record(self, checkpoint_id, commit=True):
        """

        :param checkpoint_id:
        :param commit: commits the changes to storage immediately when true.
        :return:
        """
        checkpoint_id = self.__resolve_checkpoint_id(checkpoint_id, ModelSaver.IN_GET_METHOD)
        last_checkpoint_id = self.__header[ModelSaver.H_LATEST_CHECK_POINT_ID]
        if last_checkpoint_id == checkpoint_id:
            self.__header[ModelSaver.H_LATEST_CHECK_POINT_ID] = ModelSaver.NONE_CHECKPOINT
        self.__header[ModelSaver.H_NUM_OF_CHECK_POINT] -= 1
        checkpoint_store = self.__header[ModelSaver.H_CHECK_POINT_STORE]
        del checkpoint_store[checkpoint_id]

        if commit:
            self.___save_header()

    def __save_checkpoint_record(self):
        """
        Saves the record checkpoint. The path for this storage can be resolved

        NOTE: This assumes that all the records are stored as ordinary python lists and not as numpy or any other format.
        Support for platforms must take this into consideration when designing the conversion functions
        :return:
        """
        checkpoint_id = self.__checkpoint_record[ModelSaver.H_CHECK_POINT_ID]
        checkpoint_addr = self.__get_checkpoint_addr(checkpoint_id)
        with open(checkpoint_addr, "w") as output_file:
            json.dump(self.__checkpoint_record, output_file, default=ModelSaver.__DATE_HANDLE_FOR_JSON)

    def __get_checkpoint_store(self):
        # TODO substitute this function with all the calls to the line 'self.__header[self.H_CHECK_POINT_STORE]'
        return self.__header[self.H_CHECK_POINT_STORE]

    def __get_from_checkpoint_store(self, checkpoint_id):
        temp = copy.deepcopy(self.__get_checkpoint_store()[checkpoint_id])
        temp[ModelSaver.H_CHECK_POINT_ID] = checkpoint_id
        return temp

    def delete_checkpoint(self, checkpoint_id=LAST_CHECKPOINT, commit=True):
        """

        :param commit:
        :param checkpoint_id:
        :return:
        """
        checkpoint_id = self.__resolve_checkpoint_id(checkpoint_id, ModelSaver.IN_GET_METHOD)
        checkpoint_addr = self.__get_checkpoint_addr(checkpoint_id)
        self.__delete_header_record(checkpoint_id, commit)
        os.remove(checkpoint_addr)

    def load_checkpoint(self, checkpoint_id=(LATEST_CHECK_POINT_ID, LAST_CHECKPOINT),
                        load_tensors=True, commit=True, reset=False, **args):

        checkpoint_id = self.__resolve_checkpoint_id(checkpoint_id, ModelSaver.IN_GET_METHOD)

        if reset or checkpoint_id is None:
            self.__checkpoint_record = None
            return False

        checkpoint_addr = self.__get_checkpoint_addr(checkpoint_id)
        if os.path.isfile(self.__header_address):
            with open(checkpoint_addr, 'r') as infile:
                self.__checkpoint_record = json.load(infile, object_pairs_hook=OrderedDict)
                pyprams = self.__checkpoint_record[ModelSaver.CHECK_POINT_STORE]
                self.__header[ModelSaver.H_LATEST_CHECK_POINT_ID] = checkpoint_id
                if load_tensors:
                    self.__converter_instance.set_tensors(pyprams, **args)
                if commit:
                    self.___save_header()
                return True

    def change_working_tensor_prams(self, new_tensor_prams):
        self.__converter_instance.change_tensors(new_tensor_prams)

    def create_check_point(self, checkpoint_efficiency, checkpoint_type=DYNAMIC_CHECKPOINT,
                           checkpoint_id=LAST_CHECKPOINT, **args):
        """

        :param checkpoint_efficiency:
        :param checkpoint_type: can either be static or dynamic. If it is static, the current designated checkpoint is
        replaced each time. If it is dynamic, each checkpoint creates a new record. And this continues on.
        :param checkpoint_id: only valid if checkpoint_type is STATIC_CHECKPOINT. This shows which checkpoint to use
                              for static checkpoint
        :return:

        """
        checkpoint_store = self.__converter_instance.get_pyprams(**args)
        if checkpoint_type == ModelSaver.DYNAMIC_CHECKPOINT:
            self.__new_record(checkpoint_efficiency, checkpoint_store)
        elif checkpoint_type == ModelSaver.STATIC_CHECKPOINT:
            checkpoint_id = self.__resolve_checkpoint_id(checkpoint_id, ModelSaver.NOT_IN_GET_METHOD)
            self.__new_record(checkpoint_efficiency, checkpoint_store, checkpoint_id=checkpoint_id)

    def delete_all_saved_states(self):
        checkpoint_store = self.__header[ModelSaver.H_CHECK_POINT_STORE]
        checkpoint_id_list = list(checkpoint_store.keys())
        for checkpoint_id in checkpoint_id_list:
            self.__delete_header_record(checkpoint_id, False)
            checkpoint_addr = self.__get_checkpoint_addr(checkpoint_id)
            with contextlib.suppress(FileNotFoundError):
                os.remove(checkpoint_addr)
        self.commit_all()

    def set_first_run(self):
        self.__first_run = True

    def set_not_first_run(self):
        self.__first_run = False

    def __ensure_first_run(self):
        if self.__first_run:
            self.set_not_first_run()
            return True
        return False

    def __rebase_ignore_bad_checkpoint(self, checkpoint_efficiency, old_latest_checkpoint_id, show_rebase_status,
                                       **args):

        checkpoint_store = self.__header[ModelSaver.H_CHECK_POINT_STORE]

        last_efficiency = checkpoint_store[old_latest_checkpoint_id][ModelSaver.H_CHECK_POINT_EFFICIENCY]

        if last_efficiency < checkpoint_efficiency:  # signifies that the last checkpoint is better
            # TODO here "efficiency" means "not efficiency" or "loss". This naming must be corrected:
            # Regardless if the name is changed, "efficiency" is treated as "loss". Correcting this
            # will avoid confusion.

            # at this point, the actual latest_checkpoint_id will belong to the new checkpoint. But this method attempts
            # to check if the current checkpoint had shown any significant improvement over the previous one. If no
            # improvement was shown, this gets reverted.
            self.load_checkpoint(old_latest_checkpoint_id, **args)
            if show_rebase_status:
                print("Rebasing to checkpoint: %d" % old_latest_checkpoint_id)

    def __rebase_best_checkpoint(self, new_checkpoint_efficiency, old_latest_checkpoint_id,
                                 rebase_best_checkpoint_radius, show_rebase_status,
                                 **args):

        low_range = old_latest_checkpoint_id + rebase_best_checkpoint_radius[0]
        high_range = old_latest_checkpoint_id + rebase_best_checkpoint_radius[1]

        query_result = self.query_checkpoint_info(
            check_point_id_range=(low_range, high_range),
            right_range=ModelSaver.CLOSED, left_range=ModelSaver.CLOSED)

        if query_result is None or len(query_result) == 0:
            return

        # min_ele = min(query_result, key=lambda x: x[ModelSaver.H_CHECK_POINT_EFFICIENCY])

        min_ele = self.__get_from_checkpoint_store(old_latest_checkpoint_id)
        for ele in query_result:
            if min_ele is None:
                min_ele = ele
            elif ele[ModelSaver.H_CHECK_POINT_EFFICIENCY] < min_ele[ModelSaver.H_CHECK_POINT_EFFICIENCY]:
                min_ele = ele

        if new_checkpoint_efficiency <= min_ele[ModelSaver.H_CHECK_POINT_EFFICIENCY]:
            # avoid rebasing if the most recent record has the highest efficiency.
            return

        rebase_id = min_ele[ModelSaver.H_CHECK_POINT_ID]
        self.load_checkpoint(rebase_id, **args)
        if show_rebase_status:
            print("Rebasing to checkpoint: %d" % rebase_id)

    def checkpoint_model(self, checkpoint_efficiency=None,
                         skip_type=TimeIterSkipManager.ST_ITER_SKIP,
                         skip_duration=TimeIterSkipManager.DEFAULT_SKIP,
                         checkpoint_type=DYNAMIC_CHECKPOINT,
                         checkpoint_id=(LATEST_CHECK_POINT_ID, LAST_CHECKPOINT), reset=False,
                         exec_on_checkpoint=None, exec_on_first_run=None, running_avg_efficiency=True,
                         rebase_checkpoint=REBASE_CHECKPOINT_IGNORE,
                         rebase_best_checkpoint_radius=DEFAULT_REBASE_BEST_CHECKPOINT_RADIUS,
                         show_rebase_status=True,
                         **args):
        """
        TODO: checkpoint_efficiency must be renamed to "loss" or something else signifying loss:
                `checkpoint_efficiency' will be treated that way, henceforth.

        This is called periodically to actually checkpoint the model's state
        :return: Returns True when a checkpoint is made and False otherwise.
        """

        if self.__ensure_first_run() or reset:
            checkpoint_id = self.__resolve_checkpoint_id(checkpoint_id, ModelSaver.NOT_IN_GET_METHOD)

            if checkpoint_type == ModelSaver.DYNAMIC_CHECKPOINT:
                if checkpoint_id is None:
                    # creates a checkpoint if this is the first checkpoint
                    self.create_check_point(checkpoint_efficiency=checkpoint_efficiency,
                                            checkpoint_type=checkpoint_type,
                                            checkpoint_id=checkpoint_id, **args)
                else:
                    if not self.load_checkpoint(checkpoint_id=checkpoint_id, reset=False, **args):
                        raise InvalidCheckpointID

            elif checkpoint_type == ModelSaver.STATIC_CHECKPOINT:
                if checkpoint_id is not None:
                    if not self.load_checkpoint(checkpoint_id=checkpoint_id, reset=False, **args):
                        self.create_check_point(checkpoint_efficiency=checkpoint_efficiency,
                                                checkpoint_type=checkpoint_type,
                                                checkpoint_id=checkpoint_id, **args)
                else:
                    raise InvalidCheckpointID

            self.__time_iter_skip = ModelSaver.TimeIterSkipManager(skip_type=skip_type, skip_duration=skip_duration)
            self.__running_avg_in_checkpoint_model = RunningAvg()
            if exec_on_first_run is not None:
                exec_on_first_run()

        if running_avg_efficiency and checkpoint_efficiency is not None:
            self.__running_avg_in_checkpoint_model.add_to_avg(checkpoint_efficiency)

        if self.__time_iter_skip.check_execute_checkpoint():

            checkpoint_efficiency = self.__running_avg_in_checkpoint_model.get_avg()
            self.__running_avg_in_checkpoint_model.reset()

            old_latest_checkpoint_id = None  # latest checkpoint, before the new checkpoint gets added.
            if rebase_checkpoint != ModelSaver.REBASE_CHECKPOINT_IGNORE:
                try:
                    old_latest_checkpoint_id = self.__resolve_checkpoint_id(ModelSaver.LATEST_CHECK_POINT_ID,
                                                                            ModelSaver.IN_GET_METHOD)
                except InvalidCheckpointID:
                    # signifies that there does not exist a "latest checkpoint id" at this point.
                    pass

            self.create_check_point(checkpoint_efficiency=checkpoint_efficiency, checkpoint_type=checkpoint_type,
                                    checkpoint_id=checkpoint_id, **args)

            if old_latest_checkpoint_id is not None:
                if rebase_checkpoint == ModelSaver.REBASE_IGNORE_BAD_CHECKPOINT:
                    self.__rebase_ignore_bad_checkpoint(checkpoint_efficiency, old_latest_checkpoint_id,
                                                        show_rebase_status, **args)
                elif rebase_checkpoint == ModelSaver.REBASE_BEST_CHECKPOINT:
                    self.__rebase_best_checkpoint(checkpoint_efficiency, old_latest_checkpoint_id,
                                                  rebase_best_checkpoint_radius,
                                                  show_rebase_status, **args)
            if exec_on_checkpoint is not None:
                exec_on_checkpoint()

            return True

        return False

    def query_get_last_record(self):
        checkpoint_store = self.__header[ModelSaver.H_CHECK_POINT_STORE]
        if len(checkpoint_store) == 0:
            return None
        return copy.deepcopy(ModelSaver.__last(checkpoint_store))

    def query_get_first_record(self):
        checkpoint_store = self.__header[ModelSaver.H_CHECK_POINT_STORE]
        if len(checkpoint_store) == 0:
            return None
        return copy.deepcopy(ModelSaver.__first(checkpoint_store))

    @staticmethod
    def __ensure_exactly_one_field(**args):

        counter = 0
        field_name = None
        if len(args) >= 1:
            for key, value in args.items():
                if value is not None:
                    counter += 1
                    field_name = key
            if counter == 1:
                return field_name

        raise InvalidNumberOfArgumentsPassed

    @staticmethod
    def __ensure_range(**args):

        for ele_key, ele_val in args.items():
            if (ele_val is None) or ((len(ele_val) == 2) and (ele_val[0] <= ele_val[1])):
                continue
            raise InvalidRangeArg

    def query_checkpoint_info(self, check_point_id_range=None,
                              check_point_time_range=None,
                              check_point_serial_number_range=None,
                              check_point_efficiency_range=None,
                              left_range=CLOSED,
                              right_range=CLOSED):
        """
        Out of the given parameters, only one of the parameters must be a valid range, or this method will throw an
        exception.
        :param right_range:
        :param left_range:
        :param check_point_id_range:
        :param check_point_time_range:
        :param check_point_serial_number_range: if the records are r1, r4, r7, r8, r9, r10, as an example, with the
                                                value representing the ID, the position in the listing represents the
                                                serial number. That is, it is just the record count range, starting from
                                                the beginning.
        :param check_point_efficiency_range:
        :return: the record details which can be used to load the data. This is a list of dictionaries. Each dictionary
                 contains all the attributes describing a record along with its record id, associated with their
                 respective flags.
        """
        arguments = {
            'check_point_id_range': check_point_id_range,
            'check_point_time_range': check_point_time_range,
            'check_point_serial_number_range': check_point_serial_number_range,
            'check_point_efficiency_range': check_point_efficiency_range
        }

        field_name = self.__ensure_exactly_one_field(**arguments)

        self.__ensure_range(**arguments)

        result = []
        checkpoint_store = self.__header[ModelSaver.H_CHECK_POINT_STORE]

        def compute_beg_end(checkpoint_range):
            beg = checkpoint_range[0]
            end = checkpoint_range[1]

            if left_range == ModelSaver.CLOSED and right_range == ModelSaver.CLOSED:
                def compare_indx(beg, val, end):
                    return beg <= val <= end
            elif left_range == ModelSaver.CLOSED and right_range == ModelSaver.OPEN:
                def compare_indx(beg, val, end):
                    return beg <= val < end
            elif left_range == ModelSaver.OPEN and right_range == ModelSaver.CLOSED:
                def compare_indx(beg, val, end):
                    return beg < val <= end
            elif left_range == ModelSaver.OPEN and right_range == ModelSaver.OPEN:
                def compare_indx(beg, val, end):
                    return beg < val < end
            else:
                raise InvalidLeftRightArgumentCombination

            def wrap_compare(beg, val, end):
                """
                Ensures that None is treated as false, instead of throwing an exception.
                :param beg:
                :param val:
                :param end:
                :return:
                """
                if val is None:
                    return False
                return compare_indx(beg, val, end)

            return beg, end, wrap_compare

        def save_to_result(checkpoint_id):
            #  The following are the flags corresponding to the data-values that were returned:
            #  H_CHECK_POINT_ID
            #  H_CHECK_POINT_NAME
            #  H_CHECK_POINT_TIME_STAMP
            #  H_CHECK_POINT_EFFICIENCY

            # temp = copy.deepcopy(checkpoint_store[checkpoint_id])
            # temp[ModelSaver.H_CHECK_POINT_ID] = checkpoint_id
            result.append(self.__get_from_checkpoint_store(checkpoint_id))

        def range_match_store(checkpoint_range, key=None):
            store_iterator = iter(checkpoint_store)
            beg, end, compare_indx = compute_beg_end(checkpoint_range)

            if key is None:
                def default_key(val):
                    return val

                key = default_key

            while True:
                try:
                    rec_id = next(store_iterator)
                except StopIteration:
                    break

                value = key(rec_id)  # checkpoint_store[rec_id][ModelSaver.H_CHECK_POINT_EFFICIENCY]
                if value > end:
                    break

                if compare_indx(beg, value, end):
                    save_to_result(rec_id)

        if field_name == 'check_point_id_range':
            # Note that the key is the ID, in the header for each record, stored in the header file under the
            # category ModelSaver.H_CHECK_POINT_STORE
            range_match_store(check_point_id_range)
            return result

        elif field_name == 'check_point_time_range':

            def time_key(cur_id):
                return checkpoint_store[cur_id][ModelSaver.H_CHECK_POINT_TIME_STAMP]

            range_match_store(check_point_time_range, key=time_key)
            return result

        elif field_name == 'check_point_serial_number_range':
            beg, end, compare = compute_beg_end(check_point_serial_number_range)

            items_list = checkpoint_store.items()
            i = beg

            while i < end:

                if compare(beg, i, end):
                    save_to_result(items_list[i][0])
                elif i > end:
                    break

                i += 1

            return result

        elif field_name == 'check_point_efficiency_range':

            def efficiency_key(cur_id):
                return checkpoint_store[cur_id][ModelSaver.H_CHECK_POINT_EFFICIENCY]

            range_match_store(check_point_efficiency_range, key=efficiency_key)

            result.sort(key=efficiency_key)

            return result

    def __add_header_record_to_header(self, checkpoint_id, checkpoint_name, checkpoint_timestamp,
                                      checkpoint_efficiency, commit=True):
        self.__header[ModelSaver.H_NUM_OF_CHECK_POINT] += 1
        self.__header[ModelSaver.H_LATEST_CHECK_POINT_ID] = checkpoint_id
        checkpoint_store = self.__header[ModelSaver.H_CHECK_POINT_STORE]
        new_header = self.__create_record_header_structure(checkpoint_name,
                                                           checkpoint_timestamp,
                                                           checkpoint_efficiency)
        checkpoint_store[checkpoint_id] = new_header

        if commit:
            self.___save_header()

    def __new_record(self, checkpoint_efficiency, checkpoint_store, checkpoint_id=None, commit=True):
        if checkpoint_id is None:
            checkpoint_id = self.__get_new_checkpoint_id()
        checkpoint_id = self.__resolve_checkpoint_id(checkpoint_id, ModelSaver.NOT_IN_GET_METHOD)
        checkpoint_name = self.get_checkpoint_name(checkpoint_id)
        checkpoint_timestamp = datetime.datetime.now()
        self.__checkpoint_record = self.__create_record_structure(checkpoint_id,
                                                                  checkpoint_name,
                                                                  checkpoint_timestamp,
                                                                  checkpoint_efficiency,
                                                                  checkpoint_store)
        self.__add_header_record_to_header(checkpoint_id, checkpoint_name,
                                           checkpoint_timestamp, checkpoint_efficiency, commit)
        self.__save_checkpoint_record()

    def __new_header(self, commit=True):
        self.__header = self.__create_header_structure(num_of_checkpoints=0, free_checkpoint_id=0,
                                                       latest_checkpoint_id=ModelSaver.NONE_CHECKPOINT,
                                                       checkpoint_store=OrderedDict())
        if commit:
            self.___save_header()

    def commit_all(self):
        """
        right now this method only supports late commit for header.
        :return:
        """
        self.___save_header()

    # Checkpoint name and addressing:

    def __checkpoint_num_to_str(self, check_point_id):
        """

        :param check_point_id: Checkpoint ID
        :return:
        """
        str_indx = str(check_point_id)
        for i in range(self.__check_point_digits - len(str_indx)):
            str_indx = "0" + str_indx
        return str_indx

    def __get_checkpoint_index_addr(self):
        result = self.__save_file_path
        if result[len(result) - 1] != "/":
            result += "/"
        result += self.__save_name
        result += ModelSaver.INDEX_FILE_POSTFIX
        result += self.__extension
        return result

    def __get_checkpoint_addr(self, check_point_id):
        """

        :param check_point_id: Checkpoint ID
        :return:
        """
        result = self.__save_file_path
        if result[len(result) - 1] != "/":
            result += "/"
        result += self.get_checkpoint_name(check_point_id)
        result += self.__extension
        return result

    def get_checkpoint_name(self, check_point_id):
        """

        :param check_point_id: Checkpoint ID
        :return:
        """
        result = self.__save_name
        result += self.__checkpoint_num_to_str(check_point_id)
        return result


if __name__ == "__main__":
    x = {
        "name": "John",
        "age": 30,
        "city": "New York"
    }

    y = json.dumps(x)

    print(y)

    import json

    print(json.dumps({"name": "John", "age": 30}))
    print(json.dumps(["apple", "bananas"]))
    print(json.dumps(("apple", "bananas")))
    print(json.dumps("hello"))
    print(json.dumps(42))
    print(json.dumps(31.76))
    print(json.dumps(True))
    print(json.dumps(False))
    print(json.dumps(None))

    date_handler = lambda obj: (
        obj.isoformat()
        if isinstance(obj, (datetime.datetime, datetime.date))
        else None
    )

    d = json.loads(json.dumps({"name": "John", "age": [30, 4, 5, 6], "date": datetime.datetime.now()},
                              default=date_handler))

    print(d["name"])
    print(d["age"])

    with open('data.json', 'w') as outfile:
        json.dump(d, outfile)
    data = None
    with open('data.json', 'r') as infile:
        data = json.load(infile)
        # data_obj = json.loads(data)
        print("DATE === " + str(data["date"]))

    a = {5: datetime.datetime.now(), 10: 10}
    print(json.dumps(a, default=date_handler))
