from abc import ABC

from encryption.encryption_manager import EncryptionMethod, BlockMode


class EncryptionDatastoreConstants(ABC):
    """
    Constants for the datastore to be used for encryption storage and retrieval
    """

    PLAINTEXT_TYPE = "plaintext_type"
    PLAINTEXT = "plaintext"
    ENCRYPTION_METHOD = "encryption_method"
    BLOCK_MODE = "block_mode"
    ENCRYPTION_KEY = "encryption_key"
    NONCE = "nonce"
    IV = "iv"
    CIPHERTEXT = "ciphertext"

    PLAINTEXT_TYPE_BINARY = "binary"
    PLAINTEXT_TYPE_DICT = "dict"

    ALL_FIELDS_DEF = [PLAINTEXT_TYPE, PLAINTEXT, ENCRYPTION_METHOD, BLOCK_MODE, ENCRYPTION_KEY, NONCE, IV, CIPHERTEXT]
    POSSIBLE_PLAINTEXT_TYPES = [PLAINTEXT_TYPE_BINARY, PLAINTEXT_TYPE_DICT]
    POSSIBLE_ENCRYPTION_METHODS = [EncryptionMethod.AES, EncryptionMethod.DES3, EncryptionMethod.DES,
                                   EncryptionMethod.SHIFT, EncryptionMethod.XOR]
    POSSIBLE_BLOCK_MODES = [BlockMode.ECB, BlockMode.CBC, BlockMode.CFB, BlockMode.OFB, BlockMode.CTR]
