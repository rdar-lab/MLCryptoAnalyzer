import os
from abc import abstractmethod, ABC

import numpy as np
from keras.utils import Sequence

from datastore.datastore import FileDatastore, InMemDatastore
from datastore.encryption_datastore_constants import EncryptionDatastoreConstants
from ml.training_data_generator import TrainingDataGenerator


class BaseKerasDataGenerator(Sequence, ABC):
    """
    A data generator for the Keras model training
    The data generator is responsible to generate or provide the data used for training
    The implementation needs to include  __getitem__ and __len__ implementations
    This implementation will be based on a datastore, and will bridge the gap from that format to a vectorized format
    required for ML training
    """
    _max_input_buffer_size: int
    _input_field_name: str
    _output_field_name: str
    _output_categories: list
    _scale_inputs: bool

    def __init__(self, max_input_buffer_size: int, input_field_name: str, output_field_name: str,
                 output_categories: list, scale_inputs: bool = False):
        """
        A Ctor for the BaseKerasDataGenerator class
        :param max_input_buffer_size: The maximum input size
        :param input_field_name: The name of the field which will used as input (X) data
        :param output_field_name:  The name of the field which will used as the output (Y) data
        :param output_categories: The categories list for output (for the one-hot vector for output)
        :param scale_inputs: Indication if to apply manual and fixed scaling of the data. Not required if the model for training uses BatchNorm on the input layer
        """
        self._max_input_buffer_size = max_input_buffer_size
        self._input_field_name = input_field_name
        self._output_field_name = output_field_name
        self._output_categories = output_categories
        self._scale_inputs = scale_inputs

    def _prepare_input_matrix(self, inputs_list):
        """
        Encodes the list of inputs into an NP matrix
        :param inputs_list: a list of the 'bytes' of all the training example inputs
        :return: An NP matrix of shape (TRAINING_RECORDS,MAX-INPUT-LEN)
        """
        m = len(inputs_list)
        input_matrix = np.zeros((m, self._max_input_buffer_size))
        for i in range(m):
            training_example = inputs_list[i]
            for j in range(len(training_example)):
                if j < self._max_input_buffer_size:
                    if self._scale_inputs:
                        input_matrix[i, j] = (training_example[j] - 128) / 256
                    else:
                        input_matrix[i, j] = training_example[j]
        return input_matrix

    def _prepare_output_matrix(self, target_list):
        """
        Encodes the result as a matrix of one-hot representation of the classification target of the training examples
        :param target_list: The classification targets as a list of the STR results over the possible classes
        :return: An NP matrix of shape (TRAINING_RECORDS, POSSIBLE_TARGET_CATEGORIES)
        """
        m = len(target_list)
        output_matrix = np.zeros((m, len(self._output_categories)))
        for i in range(m):
            result_x = target_list[i]
            if type(result_x) == bytes or type(result_x) == bytearray:
                result_x = str(result_x, encoding="UTF-8")
            j = self._output_categories.index(result_x)
            output_matrix[i, j] = 1
        return output_matrix

    def _convert_records(self, data):
        """
        Converts the records from the list of dict format to the format required for ML training
        :param data: A list of dict items (Each dict item represents one training example)
        :return: Inputs - an NP matrix of shape (TRAINING_RECORDS,MAX-INPUT-LEN), Outputs - nn NP matrix of shape (TRAINING_RECORDS, POSSIBLE_TARGET_CATEGORIES)
        """
        inputs = list(map(lambda record: record[self._input_field_name], data))
        inputs = self._prepare_input_matrix(inputs)

        outputs = list(map(lambda record: record[self._output_field_name], data))
        outputs = self._prepare_output_matrix(outputs)
        return inputs, outputs

    @abstractmethod
    def get_items_internal(self, index):
        """
        The internal representation of getting the items of the 'index' batch number
        :param index:
        :return:
        """
        raise NotImplementedError()

    def __getitem__(self, index):
        """
        Returns the X,Y training examples on mini-batch 'index'
        :param index: The number of the mini-batch to return
        :return: A tuple of (X,Y)
        """
        data = self.get_items_internal(index)
        records = self._convert_records(data)
        return records

    @abstractmethod
    def __len__(self):
        """
        Returns the number of mini-batches
        :return: The number of mini-batches
        """
        raise NotImplementedError()


class KerasFromFilesDataGenerator(BaseKerasDataGenerator):
    """
    An implementation of the data generator which is based on reading the data from multi-file datastore implementation
    The multi-file implementation is required in order to return the number of mini-batches without reading any of them
    to the main memory
    """
    __files_directory: str
    __file_name_prefix: str
    __file_name_postfix: str

    __num_of_files = 0

    def __init__(self, max_input_buffer_size: int,
                 input_field_name: str, output_field_name: str, output_categories: list, files_directory,
                 file_name_prefix, file_name_postfix, scale_inputs: bool = False):
        """
        Ctor for the KerasFromFileDataGenerator class
        :param max_input_buffer_size: The maximum input size
        :param input_field_name: The name of the field which will used as input (X) data
        :param output_field_name: The name of the field which will used as the output (Y) data
        :param output_categories: The categories list for output (for the one-hot vector for output)
        :param files_directory: The directory in the disk where the datastore files resides
        :param file_name_prefix: The prefix of the file names
        :param file_name_postfix: THe postfix of the file names
        :param scale_inputs:  Indication if to apply manual and fixed scaling of the data. Not required if the model for training uses BatchNorm on the input layer
        """
        super().__init__(max_input_buffer_size, input_field_name, output_field_name, output_categories,
                         scale_inputs=False)
        self.__files_directory = files_directory
        self.__file_name_prefix = file_name_prefix
        self.__file_name_postfix = file_name_postfix

        files = os.listdir(self.__files_directory)
        for file in files:
            if file.startswith(self.__file_name_prefix) and file.endswith(self.__file_name_postfix):
                self.__num_of_files += 1

    def get_items_internal(self, index):
        """
        Reads the file that represents the specific mini-batch, converts the data and returns it
        :param index: The mini-batch number
        :return: A tuple of (X,Y)
        """
        file = self.__files_directory + "/" + self.__file_name_prefix + \
               str(index + 1) + self.__file_name_postfix
        datastore = FileDatastore(EncryptionDatastoreConstants.ALL_FIELDS_DEF, file, "rb")
        data = datastore.read_all()
        datastore.close()
        return data

    def __len__(self):
        """
        Returns the number of mini-batches (which is the number of files found)
        :return:
        """
        return self.__num_of_files


class KerasOnlineDataGenerator(BaseKerasDataGenerator):
    """
    An implementation of the data generator which generates the data on the fly.
    Each time the __getitem__ will be called, a new mini-batch will be generated and returned
    Using this implementation the model will not be able to overfit to the data since fresh data is generated all the time
    """
    def __init__(self, max_input_buffer_size: int, input_field_name: str, output_field_name: str,
                 output_categories: list, plaintext_generators: dict = None, encryption_generators: dict = None,
                 batch_size: int = 100,
                 scale_inputs: bool = False,
                 min_plaintext_size: int = None,
                 max_plaintext_size: int = None
                 ):
        """
        The Ctor for the KerasOnlineDataGenerator class
        :param max_input_buffer_size: The maximum input size
        :param input_field_name: The name of the field which will used as input (X) data
        :param output_field_name: The name of the field which will used as the output (Y) data
        :param output_categories: The categories list for output (for the one-hot vector for output)
        :param plaintext_generators: a dict of format (NAME->Generator) for the plaintext generators to use
        :param encryption_generators: a dict of format ((METHOD, BMODE)->EncryptionManager) for the encryption generators to use
        :param batch_size: The batch size to create
        :param scale_inputs: Indication if to apply manual and fixed scaling of the data. Not required if the model for training uses BatchNorm on the input layer
        :param min_plaintext_size: The minimum plaintext size, If NULL a default will be used based on the max_input_buffer_size
        :param max_plaintext_size: The maximum plaintext size, If NULL a default will be used based on the max_input_buffer_size
        """
        super().__init__(max_input_buffer_size, input_field_name, output_field_name, output_categories, scale_inputs)

        self.__output = InMemDatastore(EncryptionDatastoreConstants.ALL_FIELDS_DEF)

        if min_plaintext_size is None:
            min_plaintext_size = int(max_input_buffer_size / 4)

        if max_plaintext_size is None:
            max_plaintext_size = int(max_input_buffer_size)

        self.__generator = TrainingDataGenerator(self.__output, batch_size, plaintext_generators, encryption_generators,
                                                 min_plaintext_size,
                                                 max_plaintext_size,
                                                 quiet=True)

    def get_items_internal(self, index):
        """
        Generates a batch and returns it
        :param index: The index. Not used (but expected to be 0)
        :return: A tuple of (X,Y)
        """
        self.__generator.generate_data()
        return self.__output.read_all()

    def __len__(self):
        """
        Number of mini-batches.
        Since this is generated on the fly, will always be 1.
        :return:
        """
        return 1
