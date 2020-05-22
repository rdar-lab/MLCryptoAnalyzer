from datastore.encryption_datastore_constants import EncryptionDatastoreConstants
from ml.keras_model_creator import *
from ml.ml_utils import train_and_evaluate


def trial_1_simple_plaintext_type_sanity_check():
    """
    Trial 1
    -------
    This is just a small sanity check
    Since there is no cipher the cipertext=plaintext
    The task of the ML is to recognize the difference between english text and binary data directly from the data
    A very easy task

    ML model - FC NN
    Training rounds = 50
    Plaintext = Binary / Text
    Ciphers = None

    Result = Success

    """
    evaluation_field = EncryptionDatastoreConstants.PLAINTEXT_TYPE
    possible_values = EncryptionDatastoreConstants.POSSIBLE_PLAINTEXT_TYPES
    plaintext_generators = None  # use default
    encryption_generators = {
        ("NONE", None): None
    }
    model_creator = KerasFullyConnectedNNModelCreator(len(possible_values))
    train_and_evaluate(model_creator, evaluation_field, possible_values, plaintext_generators, encryption_generators,
                       training_rounds=50)


if __name__ == '__main__':
    trial_1_simple_plaintext_type_sanity_check()
