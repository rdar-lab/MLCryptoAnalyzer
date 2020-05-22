import random
from abc import ABC, abstractmethod

from utils.dict_utils import read_dict_file


class PlaintextGenerator(ABC):
    """
    This class represents a plaintext generator. The plaintext generator is responsible for generating the plaintexts
    for the ML tests. Currently the following types are supported: Binary plaintext, Dict based plaintext
    """
    __min_data_size: int
    __max_data_size: int

    def __init__(self, min_data_size: int, max_data_size: int):
        """
        Ctor for the PlaintextGenerator class
        :param min_data_size: The min data size for the data generation
        :param max_data_size: The max data size for the data generation
        """
        if min_data_size < 0:
            raise Exception("min size cannot be negative")

        if max_data_size < 0:
            raise Exception("max size cannot be negative")

        if max_data_size < min_data_size:
            raise Exception("max size cannot be smaller then min size")

        self.__min_data_size = min_data_size
        self.__max_data_size = max_data_size

    @property
    def min_data_size(self):
        return self.__min_data_size

    @property
    def max_data_size(self):
        return self.__max_data_size

    @abstractmethod
    def generate(self):
        """
        Generates a plaintext record and returns it
        :return: The generated data in a 'bytes' data type
        """
        raise NotImplementedError()


class BinaryPlaintextGenerator(PlaintextGenerator):
    """
    A generator for random Binary data
    """
    def generate(self):
        """
        Generates a plaintext record and returns it
        :return: The generated data in a 'bytes' data type
        """
        data_size = random.randint(self.min_data_size, self.max_data_size)
        result_data: bytearray = bytearray(data_size)
        for i in range(0, data_size):
            result_data[i] = random.randint(0, 255)
        return bytes(result_data)


class DictionaryPlaintextGenerator(PlaintextGenerator):
    """
    A generator for dict based plaintext.
    Will generate a text based on random words from the dictionary
    """

    __dict_path = None
    __words = None

    def __init__(self, dict_path: str, min_data_size: int, max_data_size: int):
        """
        Ctor for the DictionaryPlaintextGenerator class
        :param dict_path: The location fot he dict file to pull the words from
        :param min_data_size: The min data size for the data generation
        :param max_data_size: The max data size for the data generation
        """
        super().__init__(min_data_size, max_data_size)
        self.__dict_path = dict_path
        self.__words, _ = read_dict_file(self.__dict_path)

    def generate(self):
        """
        Generates a plaintext record and returns it
        :return: The generated data in a 'bytes' data type
        """
        data_size = random.randint(self.min_data_size, self.max_data_size)
        result_data_str: str = ""
        while len(result_data_str) < data_size:
            random_word = random.randrange(0, len(self.__words))
            result_data_str = result_data_str + self.__words[random_word] + " "
        return bytes(result_data_str, encoding="UTF-8")
