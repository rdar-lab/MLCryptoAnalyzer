from Crypto.Random import get_random_bytes

from datastore.encryption_datastore_constants import EncryptionDatastoreConstants
from encryption.encryption_manager import EncryptionMethod, BlockMode, EncryptionManager
from generators.plaintext_generator import DictionaryPlaintextGenerator
from ml.keras_model_creator import *
from ml.ml_utils import train_and_evaluate
from utils.dict_utils import ENGLISH_DICT_PATH


def trial_5_detect_xor_or_shift_cipher_single_keys():
    """
    Trial 5
    -------
    Testing ability to recognize the type of cipher used from XOR or SHIFT ciphers when the plaintext is english text only
    The task of the ML is to recognize the cipher that was used
    Each cipher will use the same key all the time

    ML model - FC NN, 5 hidden layers of 100 hidden units per layer
    Training rounds = 50
    Plaintext = Text
    Ciphers = XOR (with single key) / SHIFT (with single key)

    Result = Success

    """
    evaluation_field = EncryptionDatastoreConstants.ENCRYPTION_METHOD
    possible_values = EncryptionDatastoreConstants.POSSIBLE_ENCRYPTION_METHODS
    plaintext_generators = {
        EncryptionDatastoreConstants.PLAINTEXT_TYPE_DICT: DictionaryPlaintextGenerator(
            ENGLISH_DICT_PATH, 1000, 1500)
    }
    encryption_generators = {
        (EncryptionMethod.XOR, BlockMode.ECB): EncryptionManager(
            EncryptionMethod.XOR,
            encryption_key=get_random_bytes(2000)),
        (EncryptionMethod.SHIFT, BlockMode.ECB): EncryptionManager(
            EncryptionMethod.SHIFT,
            encryption_key=get_random_bytes(2000))
    }
    model_creator = KerasFullyConnectedNNModelCreator(len(possible_values), units_per_layer=100, hidden_layers=5)
    train_and_evaluate(model_creator, evaluation_field, possible_values, plaintext_generators, encryption_generators,
                       training_rounds=50)


if __name__ == '__main__':
    trial_5_detect_xor_or_shift_cipher_single_keys()
