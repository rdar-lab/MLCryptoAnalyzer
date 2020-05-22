from datastore.encryption_datastore_constants import EncryptionDatastoreConstants
from encryption.encryption_manager import EncryptionMethod, BlockMode, EncryptionManager
from ml.keras_model_creator import *
from ml.ml_utils import train_and_evaluate


def trial_10_detect_plaintext_type_with_shift_short_changing_key():
    """
    Trial 10
    -------
    Testing ability to recognize the type of text when the cipher shift with short key length
    The task of the ML is to recognize the difference between english text and binary data

    ML model - FC NN
    Training rounds = 1000
    Plaintext = Binary / Text
    Ciphers = SHIFT (changing keys of short size)

    Result = Success

    """
    evaluation_field = EncryptionDatastoreConstants.PLAINTEXT_TYPE
    possible_values = EncryptionDatastoreConstants.POSSIBLE_PLAINTEXT_TYPES
    plaintext_generators = None  # use default
    encryption_generators = {
        (EncryptionMethod.SHIFT, BlockMode.ECB): EncryptionManager(
            EncryptionMethod.SHIFT,
            encryption_key_size=5)
    }
    model_creator = KerasFullyConnectedNNModelCreator(len(possible_values), hidden_layers=5)
    train_and_evaluate(model_creator, evaluation_field, possible_values, plaintext_generators, encryption_generators,
                       training_rounds=100)


if __name__ == '__main__':
    trial_10_detect_plaintext_type_with_shift_short_changing_key()
