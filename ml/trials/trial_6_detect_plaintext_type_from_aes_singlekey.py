from Crypto.Random import get_random_bytes

from datastore.encryption_datastore_constants import EncryptionDatastoreConstants
from encryption.encryption_manager import EncryptionMethod, BlockMode, EncryptionManager
from ml.keras_model_creator import *
from ml.ml_utils import train_and_evaluate


def trial_6_detect_plaintext_type_from_aes_singlekey():
    """
    Trial 6
    -------
    Testing ability to recognize the type of plaintext when the cipher is a AES with a single key
    The task of the ML is to recognize the difference between english text and binary data

    ML model - FC NN, 5 hidden layers of 100 hidden units per layer
    Training rounds = 1000
    Plaintext = Binary / Text
    Ciphers = AES (single key)

    Result = Failure

    """
    evaluation_field = EncryptionDatastoreConstants.PLAINTEXT_TYPE
    possible_values = EncryptionDatastoreConstants.POSSIBLE_PLAINTEXT_TYPES
    plaintext_generators = None  # use default
    encryption_generators = {
        (EncryptionMethod.AES, BlockMode.ECB): EncryptionManager(
            EncryptionMethod.AES,
            encryption_key=get_random_bytes(32))
    }
    model_creator = KerasFullyConnectedNNModelCreator(len(possible_values), hidden_layers=5, units_per_layer=100)
    train_and_evaluate(model_creator, evaluation_field, possible_values, plaintext_generators, encryption_generators,
                       training_rounds=1000)


if __name__ == '__main__':
    trial_6_detect_plaintext_type_from_aes_singlekey()
