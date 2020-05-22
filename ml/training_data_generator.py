import random

from datastore.datastore import BaseDatastore
from datastore.datastore import SplitFilesDatastore
from datastore.encryption_datastore_constants import EncryptionDatastoreConstants
from encryption.encryption_manager import EncryptionMethod, EncryptionManager, BlockMode
from generators.plaintext_generator import BinaryPlaintextGenerator, DictionaryPlaintextGenerator
from utils.dict_utils import ENGLISH_DICT_PATH

"""
All the supported encryption generators list
"""
ALL_ENC_GENERATORS = {
    (EncryptionMethod.AES, BlockMode.ECB): EncryptionManager(EncryptionMethod.AES, encryption_key_size=32,
                                                             block_mode=BlockMode.ECB),
    (EncryptionMethod.AES, BlockMode.CBC): EncryptionManager(EncryptionMethod.AES, encryption_key_size=32,
                                                             block_mode=BlockMode.CBC),
    (EncryptionMethod.AES, BlockMode.CFB): EncryptionManager(EncryptionMethod.AES, encryption_key_size=32,
                                                             block_mode=BlockMode.CFB),
    (EncryptionMethod.AES, BlockMode.OFB): EncryptionManager(EncryptionMethod.AES, encryption_key_size=32,
                                                             block_mode=BlockMode.OFB),
    (EncryptionMethod.AES, BlockMode.CTR): EncryptionManager(EncryptionMethod.AES, encryption_key_size=32,
                                                             block_mode=BlockMode.CTR),
    (EncryptionMethod.DES3, BlockMode.ECB): EncryptionManager(EncryptionMethod.DES3, encryption_key_size=24,
                                                              block_mode=BlockMode.ECB),
    (EncryptionMethod.DES3, BlockMode.CBC): EncryptionManager(EncryptionMethod.DES3, encryption_key_size=24,
                                                              block_mode=BlockMode.CBC),
    (EncryptionMethod.DES3, BlockMode.CFB): EncryptionManager(EncryptionMethod.DES3, encryption_key_size=24,
                                                              block_mode=BlockMode.CFB),
    (EncryptionMethod.DES3, BlockMode.OFB): EncryptionManager(EncryptionMethod.DES3, encryption_key_size=24,
                                                              block_mode=BlockMode.OFB),
    (EncryptionMethod.DES, BlockMode.ECB): EncryptionManager(EncryptionMethod.DES, encryption_key_size=8,
                                                             block_mode=BlockMode.ECB),
    (EncryptionMethod.DES, BlockMode.CBC): EncryptionManager(EncryptionMethod.DES, encryption_key_size=8,
                                                             block_mode=BlockMode.CBC),
    (EncryptionMethod.DES, BlockMode.CFB): EncryptionManager(EncryptionMethod.DES, encryption_key_size=8,
                                                             block_mode=BlockMode.CFB),
    (EncryptionMethod.DES, BlockMode.OFB): EncryptionManager(EncryptionMethod.DES, encryption_key_size=8,
                                                             block_mode=BlockMode.OFB),
    (EncryptionMethod.SHIFT, BlockMode.ECB): EncryptionManager(EncryptionMethod.SHIFT, encryption_key_size=8,
                                                               block_mode=BlockMode.ECB),
    (EncryptionMethod.XOR, BlockMode.ECB): EncryptionManager(EncryptionMethod.XOR, encryption_key_size=8,
                                                             block_mode=BlockMode.ECB)
}

"""
AES generators
"""
ALL_AES_GENERATORS = {
    (EncryptionMethod.AES, BlockMode.ECB): EncryptionManager(EncryptionMethod.AES, encryption_key_size=32,
                                                             block_mode=BlockMode.ECB),
    (EncryptionMethod.AES, BlockMode.CBC): EncryptionManager(EncryptionMethod.AES, encryption_key_size=32,
                                                             block_mode=BlockMode.CBC),
    (EncryptionMethod.AES, BlockMode.CFB): EncryptionManager(EncryptionMethod.AES, encryption_key_size=32,
                                                             block_mode=BlockMode.CFB),
    (EncryptionMethod.AES, BlockMode.OFB): EncryptionManager(EncryptionMethod.AES, encryption_key_size=32,
                                                             block_mode=BlockMode.OFB),
    (EncryptionMethod.AES, BlockMode.CTR): EncryptionManager(EncryptionMethod.AES, encryption_key_size=32,
                                                             block_mode=BlockMode.CTR)
}

"""
3-DES generators
"""
ALL_DES3_GENERATORS = {
    (EncryptionMethod.DES3, BlockMode.ECB): EncryptionManager(EncryptionMethod.DES3, encryption_key_size=24,
                                                              block_mode=BlockMode.ECB),
    (EncryptionMethod.DES3, BlockMode.CBC): EncryptionManager(EncryptionMethod.DES3, encryption_key_size=24,
                                                              block_mode=BlockMode.CBC),
    (EncryptionMethod.DES3, BlockMode.CFB): EncryptionManager(EncryptionMethod.DES3, encryption_key_size=24,
                                                              block_mode=BlockMode.CFB),
    (EncryptionMethod.DES3, BlockMode.OFB): EncryptionManager(EncryptionMethod.DES3, encryption_key_size=24,
                                                              block_mode=BlockMode.OFB)
}

"""
DES generators
"""
ALL_DES_GENERATORS = {
    (EncryptionMethod.DES, BlockMode.ECB): EncryptionManager(EncryptionMethod.DES, encryption_key_size=8,
                                                             block_mode=BlockMode.ECB),
    (EncryptionMethod.DES, BlockMode.CBC): EncryptionManager(EncryptionMethod.DES, encryption_key_size=8,
                                                             block_mode=BlockMode.CBC),
    (EncryptionMethod.DES, BlockMode.CFB): EncryptionManager(EncryptionMethod.DES, encryption_key_size=8,
                                                             block_mode=BlockMode.CFB),
    (EncryptionMethod.DES, BlockMode.OFB): EncryptionManager(EncryptionMethod.DES, encryption_key_size=8,
                                                             block_mode=BlockMode.OFB)
}

"""
ECB Generators
"""
ECB_GENERATORS = {
    (EncryptionMethod.AES, BlockMode.ECB): EncryptionManager(EncryptionMethod.AES, encryption_key_size=32,
                                                             block_mode=BlockMode.ECB),
    (EncryptionMethod.DES3, BlockMode.ECB): EncryptionManager(EncryptionMethod.DES3, encryption_key_size=24,
                                                              block_mode=BlockMode.ECB),
    (EncryptionMethod.DES, BlockMode.ECB): EncryptionManager(EncryptionMethod.DES, encryption_key_size=8,
                                                             block_mode=BlockMode.ECB),
    (EncryptionMethod.SHIFT, BlockMode.ECB): EncryptionManager(EncryptionMethod.SHIFT, encryption_key_size=8,
                                                               block_mode=BlockMode.ECB),
    (EncryptionMethod.XOR, BlockMode.ECB): EncryptionManager(EncryptionMethod.XOR, encryption_key_size=8,
                                                             block_mode=BlockMode.ECB)
}


class TrainingDataGenerator:
    """
    This class is responsible for generating training and evaluation data based on the plaintext generator
    and encryption managers. The generation is random
    """
    __encryption_generators: list = None
    __plaintext_generators: list = None
    __num_of_records: int = None
    __output_datastore: BaseDatastore = None
    __quiet = None

    def __init__(self, output_datastore: BaseDatastore, num_of_records: int, plaintext_generators: dict = None,
                 encryption_generators: dict = None, min_plaintext_size: int = 1000, max_plaintext_size: int = 3000,
                 quiet: bool = False):
        """
        CTOR for the TrainingDataGenerator class
        :param output_datastore: The datastore to output the generation data to
        :param num_of_records: The number of records to generate
        :param plaintext_generators: a dict of format (NAME->Generator) for the plaintext generators to use
        :param encryption_generators: a dict of format ((METHOD, BMODE)->EncryptionManager) for the encryption generators to use
        :param min_plaintext_size: The minimum plaintext size to generate
        :param max_plaintext_size: The maximum plaintext size to generate
        :param quiet: Indication if not to print the progress to the console
        """

        if plaintext_generators is None:
            plaintext_generators = {
                EncryptionDatastoreConstants.PLAINTEXT_TYPE_BINARY: BinaryPlaintextGenerator(
                    min_plaintext_size, max_plaintext_size),
                EncryptionDatastoreConstants.PLAINTEXT_TYPE_DICT: DictionaryPlaintextGenerator(
                    ENGLISH_DICT_PATH, min_plaintext_size, max_plaintext_size)
            }

        if encryption_generators is None:
            encryption_generators = ALL_ENC_GENERATORS

        self.__encryption_generators = list(encryption_generators.items())
        self.__plaintext_generators = list(plaintext_generators.items())
        self.__num_of_records = num_of_records
        self.__output_datastore = output_datastore
        self.__quiet = quiet

    def generate_data(self):
        """"
        Generate the data
        """

        if not self.__quiet:
            print("Generating data->")

        for i in range(0, self.__num_of_records):
            plaintext_type_index: int = random.randrange(0, len(self.__plaintext_generators))
            plaintext_type, plaintext_generator = self.__plaintext_generators[plaintext_type_index]
            plaintext = plaintext_generator.generate()

            encryption_generator_index: int = random.randrange(0, len(self.__encryption_generators))
            (encryption_method, block_mode), encryption_manager = self.__encryption_generators[
                encryption_generator_index]

            encryption_key = ciphertext = nonce = iv = None
            if encryption_manager is not None:
                encryption_key = encryption_manager.get_key()
                ciphertext, nonce, iv = encryption_manager.encrypt(plaintext)
            else:
                ciphertext = plaintext

            self.__output_datastore.write_record(
                {
                    EncryptionDatastoreConstants.PLAINTEXT_TYPE: plaintext_type,
                    EncryptionDatastoreConstants.PLAINTEXT: plaintext,
                    EncryptionDatastoreConstants.ENCRYPTION_METHOD: encryption_method,
                    EncryptionDatastoreConstants.BLOCK_MODE: block_mode,
                    EncryptionDatastoreConstants.ENCRYPTION_KEY: encryption_key,
                    EncryptionDatastoreConstants.NONCE: nonce,
                    EncryptionDatastoreConstants.IV: iv,
                    EncryptionDatastoreConstants.CIPHERTEXT: ciphertext
                }
            )

            if not self.__quiet:
                if i % 100 == 0:
                    print('')
                    print(str(i), end='')

                print(".", end='')


if __name__ == '__main__':
    """
    Generates training data to disk files. Can be used later with the KerasFromFilesDataGenerator 
    """
    output = SplitFilesDatastore(EncryptionDatastoreConstants.ALL_FIELDS_DEF, 100, "../resources/train", "train_",
                                 ".bin")
    generator = TrainingDataGenerator(output, 100)
    generator.generate_data()
    output.close()
