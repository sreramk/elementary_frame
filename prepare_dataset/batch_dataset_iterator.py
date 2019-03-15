# copyright (c) 2019 K Sreram, All rights reserved.
from prepare_dataset.data_set_buffer import DataBuffer


class BatchDataSetIterator:

    def __init__(self, buffer: DataBuffer, batch_size):

        self.__batches_input = None
        self.__batches_output = None

        self.__batch_size = batch_size

        self.__buffer = buffer
        self.__batch_counter = None
        self.reinitialize()

    def reinitialize(self):

        self.__batches_input = []
        self.__batches_output = []

        self.__buffer.initialize_get_random_unique()

        unique_ele = self.__buffer.get_random_unique()

        count = 0
        unit_batch_input = []
        unit_batch_output= []
        while unique_ele is not None:
            unit_batch_input.append(unique_ele[DataBuffer.DpType.INPUT_DP])
            unit_batch_output.append(unique_ele[DataBuffer.DpType.OUTPUT_DP])
            if count >= self.__batch_size:
                self.__batches_input.append(unit_batch_input)
                self.__batches_output.append(unit_batch_output)
                unit_batch_input = []
                unit_batch_output= []
                count = 0
            unique_ele = self.__buffer.get_random_unique()
            count += 1

        self.__batch_counter = 0

    def get_next_batch(self):

        if self.__batch_counter >= len(self.__batches_input):
            return None

        input_batch = self.__batches_input[self.__batch_counter]
        output_batch = self.__batches_output[self.__batch_counter]

        self.__batch_counter += 1

        return DataBuffer.create_input_output_dp(input_batch, output_batch)

    def reset_batch_iterator(self):
        self.__batch_counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        result = self.get_next_batch()
        if result is None:
            raise StopIteration
        return result

    def __len__(self):
        return len(self.__batches_input)


if __name__ == '__main__':
    def main_fnc():

        class IterationCount:

            def __init__(self, first, last):
                self.__first = first
                self.__last = last
                self.__cur_count = self.__first

            def __iter__(self):
                return self

            def __next__(self):
                if self.__cur_count < self.__last:
                    result = self.__cur_count
                    self.__cur_count += 1
                    return result

                else:
                    raise StopIteration

            def reset_iterator(self):
                self.__cur_count = self.__first

        for i in IterationCount(50, 100):
            print(i)


    main_fnc()
