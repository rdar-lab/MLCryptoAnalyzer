from Crypto.Random import get_random_bytes

from datastore.encryption_datastore_constants import EncryptionDatastoreConstants
from encryption.encryption_manager import EncryptionMethod, BlockMode, EncryptionManager
from ml.keras_model_creator import *
from ml.keras_data_generator import KerasOnlineDataGenerator
from ml.ml_utils import evaluate_model, create_and_train


def trial_11_rnn_detect_plaintext_type_from_shift_cipher_singlekey():
    """
    Trial 11
    -------
    Testing ability to recognize the type of text when the cipher is a simple SHIFT with a single key
    The task of the ML is to recognize the difference between english text and binary data
    This is the same as Trial 3 only with RNN so supports variable size of input

    ML model - RNN
    Training rounds = 100
    Plaintext = Binary / Text
    Ciphers = XOR (single key)

    Result = Success

    """

    evaluation_field = EncryptionDatastoreConstants.PLAINTEXT_TYPE
    possible_values = EncryptionDatastoreConstants.POSSIBLE_PLAINTEXT_TYPES
    plaintext_generators = None  # use default
    key = get_random_bytes(32)
    encryption_generators = {
        (EncryptionMethod.SHIFT, BlockMode.ECB): EncryptionManager(
            EncryptionMethod.SHIFT,
            encryption_key=key)
    }

    model_creator = KerasRNNModelCreator(len(possible_values), units_per_layer=100, dropout_rate=0.05)
    generator = KerasOnlineDataGenerator(10000,
                                       EncryptionDatastoreConstants.CIPHERTEXT,
                                       evaluation_field,
                                       possible_values,
                                       plaintext_generators=plaintext_generators,
                                       encryption_generators=encryption_generators,
                                       batch_size=100,
                                       min_plaintext_size=100,
                                       max_plaintext_size=10000
                                       )

    model = create_and_train(model_creator, generator, training_rounds=100)

    print("Testing with short input:")
    generator = KerasOnlineDataGenerator(500,
                                       EncryptionDatastoreConstants.CIPHERTEXT,
                                       evaluation_field,
                                       possible_values,
                                       plaintext_generators=plaintext_generators,
                                       encryption_generators=encryption_generators,
                                       batch_size=10
                                       )
    evaluation_data_x, evaluation_data_y = generator[0]
    evaluate_model(model, evaluation_data_x, evaluation_data_y)

    print("Testing with medium size inputs:")
    generator = KerasOnlineDataGenerator(4000,
                                       EncryptionDatastoreConstants.CIPHERTEXT,
                                       evaluation_field,
                                       possible_values,
                                       plaintext_generators=plaintext_generators,
                                       encryption_generators=encryption_generators,
                                       batch_size=10
                                       )
    evaluation_data_x, evaluation_data_y = generator[0]
    evaluate_model(model, evaluation_data_x, evaluation_data_y)

    print("Testing with large size inputs:")
    generator = KerasOnlineDataGenerator(10000,
                                       EncryptionDatastoreConstants.CIPHERTEXT,
                                       evaluation_field,
                                       possible_values,
                                       plaintext_generators=plaintext_generators,
                                       encryption_generators=encryption_generators,
                                       batch_size=10
                                       )
    evaluation_data_x, evaluation_data_y = generator[0]
    evaluate_model(model, evaluation_data_x, evaluation_data_y)


if __name__ == '__main__':
    trial_11_rnn_detect_plaintext_type_from_shift_cipher_singlekey()
