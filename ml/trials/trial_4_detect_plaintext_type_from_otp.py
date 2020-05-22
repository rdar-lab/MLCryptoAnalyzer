from datastore.encryption_datastore_constants import EncryptionDatastoreConstants
from encryption.encryption_manager import EncryptionMethod, BlockMode, EncryptionManager
from ml.keras_model_creator import *
from ml.ml_utils import train_and_evaluate


def trial_4_detect_plaintext_type_from_otp():
    """
    Trial 4
    -------
    Testing ability to recognize the type of text when the cipher is a real OTP (random key for each specimen)
    The task of the ML is to recognize the difference between english text and binary data
    Proven to be impossible when the keys are single use and true random, so we expect this to fail

    ML model - FC NN
    Training rounds = 50
    Plaintext = Binary / Text
    Ciphers = OTP (XOR with random key for each record)

    Result = Failure

    """
    evaluation_field = EncryptionDatastoreConstants.PLAINTEXT_TYPE
    possible_values = EncryptionDatastoreConstants.POSSIBLE_PLAINTEXT_TYPES
    plaintext_generators = None  # use default
    encryption_generators = {
        (EncryptionMethod.XOR, BlockMode.ECB): EncryptionManager(
            EncryptionMethod.XOR,
            encryption_key_size=2000)
    }
    model_creator = KerasFullyConnectedNNModelCreator(len(possible_values))
    train_and_evaluate(model_creator, evaluation_field, possible_values, plaintext_generators, encryption_generators)


if __name__ == '__main__':
    trial_4_detect_plaintext_type_from_otp()
