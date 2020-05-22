import queue
import sys
from abc import ABC, abstractmethod


class BaseDatastore(ABC):
    """
    Abstract parent for all the datastore classes
    A datastore can read and write records data
    """
    object_fields = None

    def __init__(self, object_fields: list):
        """
        Ctor for the BaseDatastore class
        :param object_fields: A list object of all the field names that datastore holds
        """
        self.object_fields = object_fields

    def write_record(self, record_data: dict):
        """
        Write a single record to the store
        :param record_data: The record to write
        :return:
        """
        raise NotImplementedError()

    def read_next_record(self):
        """
        Reads the next available record in the store
        :return: If a record is available it will return it, otherwise None will be returned
        """
        raise NotImplementedError()

    def read_all(self):
        """
        Reads all the records in the store
        :return: A list of all the records
        """
        results = []
        next_record = self.read_next_record()
        while next_record is not None:
            results.append(next_record)
            next_record = self.read_next_record()

        return results


class BinaryDatastore(BaseDatastore, ABC):
    """
    A datastore implementation ready for binary serialization (e.g. to write to binary file or to send over network)
    The format for each record is:
    [RECORD LEN] [FIELD1 LEN] [FIELD1 DATA] [FIELD2 LEN] [FIELD2 DATA]...
    Maximum record size is 1MB
    Each length header is 4 bytes
    """
    __MAX_RECORD_SIZE = 1024 * 1024
    __LEN_FIELD_SIZE = 4

    def __init__(self, object_fields: list):
        """
        A Ctor for the BinaryDatastore class
        :param object_fields: A list object of all the field names that datastore holds
        """
        super().__init__(object_fields)

    def __bin_to_record_data(self, bin_data: bytes):
        """
        An internal method responsible for conversion of the data from binary format to in mem object
        :param bin_data: The binary format
        :return: A dictionary of the field->value entries
        """
        result = {}
        index: int = 0

        for field_name in self.object_fields:
            field_size = int.from_bytes(bin_data[index:index + BinaryDatastore.__LEN_FIELD_SIZE], sys.byteorder)
            index += BinaryDatastore.__LEN_FIELD_SIZE
            if field_size > 0:
                field_data = bin_data[index:index + field_size]
                result[field_name] = field_data
                index += field_size

        return result

    def __record_data_to_bin(self, record_data: dict):
        """
        An internal method responsible for conversion of the data from in mem object (dict) format to binary format
        :param record_data: A dict format of the field->value entries
        :return: A binary array (immutable bytes type)
        """
        result = bytearray(BinaryDatastore.__MAX_RECORD_SIZE)

        index: int = 0

        for field_name in self.object_fields:
            field_data = None
            if field_name in record_data:
                field_data = record_data[field_name]

            if field_data is not None:
                if type(field_data) == str:
                    field_data = bytes(field_data, encoding="UTF-8")
                elif not type(field_data) == bytes:
                    raise Exception("Field " + field_name + " - Not supported data type =" + str(type(field_data)))

                field_size = len(field_data)
                len_bytes = field_size.to_bytes(BinaryDatastore.__LEN_FIELD_SIZE, sys.byteorder)
                result[index:index + BinaryDatastore.__LEN_FIELD_SIZE] = len_bytes
                index += BinaryDatastore.__LEN_FIELD_SIZE
                result[index:index + field_size] = field_data
                index += field_size
            else:
                field_size = 0
                len_bytes = field_size.to_bytes(BinaryDatastore.__LEN_FIELD_SIZE, sys.byteorder)
                result[index:index + BinaryDatastore.__LEN_FIELD_SIZE] = len_bytes
                index += BinaryDatastore.__LEN_FIELD_SIZE

        return bytes(result[0:index])

    @abstractmethod
    def write_internal(self, data: bytes):
        """
        Writes the binary data to the medium
        :param data: the binary data (bytes)
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def read_internal(self, read_len: int):
        """
        Read data from the medium and returns it.
        If no data is available (due to EOF for example) None should be returned
        :param read_len: size of bytes to read
        :return: The data that was read, or None if no data is available
        """
        raise NotImplementedError()

    def write_record(self, record_data: dict):
        """
        Write a single record to the store
        :param record_data: The record to write
        :return:
        """
        record_bin_data: bytes = self.__record_data_to_bin(record_data)
        record_size = len(record_bin_data)
        record_size_bytes = record_size.to_bytes(BinaryDatastore.__LEN_FIELD_SIZE, sys.byteorder)
        self.write_internal(record_size_bytes)
        self.write_internal(record_bin_data)

    def read_next_record(self):
        """
        Reads the next available record in the store
        :return: If a record is available it will return it, otherwise None will be returned
        """
        record_size_bytes = self.read_internal(BinaryDatastore.__LEN_FIELD_SIZE)

        if record_size_bytes is None or len(record_size_bytes) == 0:
            return None

        if len(record_size_bytes) < BinaryDatastore.__LEN_FIELD_SIZE:
            raise Exception(
                "Data file corrupted. Expected size header size = " + str(
                    BinaryDatastore.__LEN_FIELD_SIZE) + ", instead got = " + str(
                    len(record_size_bytes)))

        record_size = int.from_bytes(record_size_bytes, sys.byteorder)
        record_bin_data = self.read_internal(record_size)

        if record_bin_data is None:
            raise Exception("Data file corrupted. Unexpected EOF detected")

        if len(record_bin_data) != record_size:
            raise Exception(
                "Data file corrupted. Expected record size = " + str(record_size) + ", instead got = " + str(
                    len(record_bin_data)))

        record_data = self.__bin_to_record_data(record_bin_data)

        return record_data


class FileDatastore(BinaryDatastore):
    """
    Implementation of binary datastore with saving to the disk (file) capability
    """
    __file_handle = None
    __autoflush_num_records = None
    __num_records = 0

    def __init__(self, object_fields: list, file_path: str, file_mode: str, autoflush_num_records: int = 0):
        """
        Ctor for the FileDatastore class
        :param object_fields: A list object of all the field names that datastore holds
        :param file_path: The file path to write
        :param file_mode: The file mode for opening the file. Recommended to use 'wb' for writing or 'rb' for reading
        :param autoflush_num_records: Number of records to auto-flush afterwards
        """
        super().__init__(object_fields)
        self.__autoflush_num_records = autoflush_num_records
        self.__file_handle = open(file_path, mode=file_mode)

    def write_internal(self, data: bytes):
        """
         Writes the binary data to the medium
         :param data: the binary data (bytes)
         :return:
         """
        self.__file_handle.write(data)
        self.__num_records += 1

        if self.__autoflush_num_records > 0 and self.__num_records % self.__autoflush_num_records == 0:
            self.flush()

    def read_internal(self, read_len: int):
        """
        Read data from the medium and returns it.
        If no data is available (due to EOF for example) None should be returned
        :param read_len: size of bytes to read
        :return: The data that was read, or None if no data is available
        """
        data: bytearray = bytearray()
        while len(data) < read_len:
            chunk = self.__file_handle.read(read_len - len(data))

            # check EOF
            if not chunk:
                return data

            data = data + bytearray(chunk)
        return data

    def flush(self):
        """
        Manual flushing operation, to flush the file
        :return:
        """
        self.__file_handle.flush()

    def close(self):
        """
        A close method to close the file. After this is called this datastore object cannot be used anymore
        :return:
        """
        self.__file_handle.close()


class InMemDatastore(BaseDatastore):
    """
    A datastore implementation storing all the data in memory.
    The storage is on a FIFO based queue
    """
    __num_of_records_left = 0
    __data: queue.Queue = queue.Queue()

    def has_more_records(self):
        """
        Checks if there are records available
        :return: True if there are records available for reading
        """
        return self.__num_of_records_left > 0

    def pending_num_of_records(self):
        """
        Returns the number of unread records
        :return: Returns the number of unread records
        """
        return self.__num_of_records_left

    def write_record(self, record_data: dict):
        """
        Write a single record to the store
        :param record_data: The record to write
        :return:
        """
        self.__data.put(record_data)
        self.__num_of_records_left += 1

    def read_next_record(self):
        """
        Reads the next available record in the store
        :return: If a record is available it will return it, otherwise None will be returned
        """
        if self.__num_of_records_left == 0:
            return None

        next_record = self.__data.get()
        self.__num_of_records_left -= 1
        return next_record


class SplitFilesDatastore(BaseDatastore):
    """
    A special implementation of the datastore for splitting the storage to multiple files.
    This implementation supports only writing at this point. A FileDatastore can be used to read a specific chunk
    """
    __num_of_records_pre_file: int
    __files_directory: str
    __file_name_prefix: str
    __file_name_postfix: str

    __curr_file_datastore: FileDatastore = None
    __curr_index = 0

    def __init__(self, object_fields: list,
                 num_of_records_per_file: int,
                 files_directory: str,
                 file_name_prefix: str,
                 file_name_postfix: str):
        """
        A Ctor for the SplitFilesDatastore class
        :param object_fields: A list object of all the field names that datastore holds
        :param num_of_records_per_file: Number of records per file (per chunk)
        :param files_directory: The director to store the files in
        :param file_name_prefix: The file name prefix
        :param file_name_postfix: The file name postfix (usually .[extension])
        """
        super().__init__(object_fields)
        self.__num_of_records_pre_file = num_of_records_per_file
        self.__files_directory = files_directory
        self.__file_name_prefix = file_name_prefix
        self.__file_name_postfix = file_name_postfix

    def write_record(self, record_data: dict):
        """
        Write a single record to the store
        :param record_data: The record to write
        :return:
        """
        if self.__curr_index % self.__num_of_records_pre_file == 0:
            if self.__curr_file_datastore is not None:
                self.__curr_file_datastore.close()

            curr_batch = int(1 + self.__curr_index / self.__num_of_records_pre_file)
            new_name = self.__files_directory + "/" + self.__file_name_prefix + str(
                curr_batch) + self.__file_name_postfix
            self.__curr_file_datastore = FileDatastore(self.object_fields,
                                                       new_name,
                                                       "wb"
                                                       )
        self.__curr_file_datastore.write_record(record_data)
        self.__curr_index += 1

    def read_next_record(self):
        """
        Not supported in this implementation, will raise a NotImplementedError
        :return:
        """
        raise NotImplementedError("Not supported for SplitFilesDatastore")

    def close(self):
        """
        Closes the last file chunk. Should always be used to finish the write of the last file
        :return:
        """
        if self.__curr_file_datastore is not None:
            self.__curr_file_datastore.close()
