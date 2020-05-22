from Crypto.Random import get_random_bytes

from datastore.encryption_datastore_constants import EncryptionDatastoreConstants
from encryption.encryption_manager import EncryptionMethod, BlockMode, EncryptionManager
from ml.keras_model_creator import *
from ml.ml_utils import train_and_evaluate


def trial_2_detect_plaintext_type_from_shift_cipher_singlekey():
    """
    Trial 2
    -------
    Testing ability to recognize the type of text when the cipher is a legacy SHIFT (similar to Caesar cipher) with a single key
    The task of the ML is to recognize the difference between english text and binary data

    ML model - FC NN
    Training rounds = 50
    Plaintext = Binary / Text
    Ciphers = SHIFT (single key)

    Result = Success

    """
    evaluation_field = EncryptionDatastoreConstants.PLAINTEXT_TYPE
    possible_values = EncryptionDatastoreConstants.POSSIBLE_PLAINTEXT_TYPES
    plaintext_generators = None  # use default
    encryption_generators = {
        (EncryptionMethod.SHIFT, BlockMode.ECB): EncryptionManager(
            EncryptionMethod.SHIFT,
            encryption_key=get_random_bytes(2000))
    }
    model_creator = KerasFullyConnectedNNModelCreator(len(possible_values))
    train_and_evaluate(model_creator, evaluation_field, possible_values, plaintext_generators, encryption_generators)


if __name__ == '__main__':
    trial_2_detect_plaintext_type_from_shift_cipher_singlekey()
