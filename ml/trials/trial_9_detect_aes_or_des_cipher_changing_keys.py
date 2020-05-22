from datastore.encryption_datastore_constants import EncryptionDatastoreConstants
from encryption.encryption_manager import EncryptionMethod, BlockMode, EncryptionManager
from generators.plaintext_generator import DictionaryPlaintextGenerator
from ml.keras_model_creator import *
from ml.ml_utils import train_and_evaluate
from utils.dict_utils import ENGLISH_DICT_PATH


def trial_9_detect_aes_or_des_cipher_changing_keys():
    """
    Trial 9
    -------
    Testing ability to recognize the type of cipher used from AES or DES ciphers when the plaintext is english text only
    The task of the ML is to recognize the cipher that was used
    Each cipher will use different keys for every encryption operation
    Expected to fail

    ML model - FC NN, 5 hidden layers of 100 hidden units per layer
    Training rounds = 1000
    Plaintext = Text
    Ciphers = DES (with changing keys) / AES (with changing keys)

    Result = Failure

    """
    evaluation_field = EncryptionDatastoreConstants.ENCRYPTION_METHOD
    possible_values = EncryptionDatastoreConstants.POSSIBLE_ENCRYPTION_METHODS
    plaintext_generators = {
        EncryptionDatastoreConstants.PLAINTEXT_TYPE_DICT: DictionaryPlaintextGenerator(
            ENGLISH_DICT_PATH, 1000, 1500)
    }
    encryption_generators = {
        (EncryptionMethod.DES, BlockMode.ECB): EncryptionManager(
            EncryptionMethod.DES,
            encryption_key_size=8),
        (EncryptionMethod.AES, BlockMode.ECB): EncryptionManager(
            EncryptionMethod.AES,
            encryption_key_size=32)
    }
    model_creator = KerasFullyConnectedNNModelCreator(len(possible_values), units_per_layer=100, hidden_layers=5)
    train_and_evaluate(model_creator, evaluation_field, possible_values, plaintext_generators, encryption_generators,
                       training_rounds=1000)


if __name__ == '__main__':
    trial_9_detect_aes_or_des_cipher_changing_keys()
