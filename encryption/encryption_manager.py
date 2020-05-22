from abc import ABC

from Crypto.Cipher import AES, DES
from Crypto.Cipher import DES3
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad


class EncryptionMethod(ABC):
    """
    Available encryption methods
    """

    AES = "AES"
    DES3 = "DES3"
    DES = "DES"
    SHIFT = "SHIFT"
    XOR = "XOR"


class BlockMode(ABC):
    """
    Available block modes for the encryption
    """
    ECB = "ECB"
    CBC = "CBC"
    CFB = "CFB"
    OFB = "OFB"
    CTR = "CTR"


_block_mode_dict = {
    BlockMode.ECB: 1,
    BlockMode.CBC: 2,
    BlockMode.CFB: 3,
    BlockMode.OFB: 5,
    BlockMode.CTR: 6,
}


class SimpleShiftCipher:
    """
    A simple 'Caesar cipher' like cipher for binary data
    """
    __key: bytes

    def __init__(self, key: bytes):
        self.__key = key

    def encrypt(self, plaintext: bytes):
        result: bytearray = bytearray(len(plaintext))
        for index in range(len(plaintext)):
            result[index] = (plaintext[index] + self.__key[index % len(self.__key)]) & 0xFF
        return bytes(result)

    def decrypt(self, ciphertext: bytes):
        result: bytearray = bytearray(len(ciphertext))
        for index in range(len(ciphertext)):
            result[index] = (ciphertext[index] - self.__key[index % len(self.__key)]) & 0xFF
        return bytes(result)


class SimpleXorCipher:
    """
    A simple 'XOR cipher' like OTP (in case the keys are used one time only)
    """
    __key: bytes

    def __init__(self, key: bytes):
        self.__key = key

    def encrypt(self, plaintext: bytes):
        result: bytearray = bytearray(len(plaintext))
        for index in range(len(plaintext)):
            result[index] = plaintext[index] ^ self.__key[index % len(self.__key)]
        return bytes(result)

    def decrypt(self, ciphertext: bytes):
        result: bytearray = bytearray(len(ciphertext))
        for index in range(len(ciphertext)):
            result[index] = ciphertext[index] ^ self.__key[index % len(self.__key)]
        return bytes(result)


class EncryptionManager:
    """
    This class encapsulates all the required logic for symmetric encryption and decryption for this project
    Supported encryption are: Aes, Des and 3-Des (called DES3 due to variable naming issues in python)
    """
    __encryption_method: str = None
    __block_mode: str = None
    __encryption_key: bytes = None
    __block_size: int = None
    __encryption_key_size: int = None
    __randomize_key_on_every_encryption = None

    def __init__(self, encryption_method: str, encryption_key_size: int = 32, encryption_key: bytes = None,
                 block_size: int = 32, block_mode: str = BlockMode.ECB):
        """
        Ctor for the EncryptionManager class
        :param encryption_method: The encryption method to use
        :param encryption_key_size: The key size to use (needed in case generate_random_key is called afterwards)
        :param encryption_key: Needed if a specific encryption key is to be used
        :param block_size: The block size for the encryption (default 32). Will be used for padding
        :param block_mode: The block mode to use (ECB, CBC, CFB, OFB, CTR)
        """
        self.__encryption_method = encryption_method
        self.__encryption_key_size = encryption_key_size
        self.__encryption_key = encryption_key
        self.__block_size = block_size
        self.__block_mode = block_mode

        if self.__encryption_key is None:
            self.__randomize_key_on_every_encryption = True
        else:
            self.__randomize_key_on_every_encryption = False

        # Generate the next key to be used
        if self.__randomize_key_on_every_encryption:
            self.__encryption_key = get_random_bytes(self.__encryption_key_size)

    def encrypt(self, plaintext: bytes):
        """
        Encrypts the plaintext
        :param plaintext: The plaintext to encrypt
        :return: A tuple of (ciphertext, nonce, iv). The ciphertext is the encrypted result. nonce and iv may be
                 required for the decryption process if those were used in the selected  block mode
        """
        if self.__encryption_key is None:
            raise Exception("Encryption key not set")

        cipher = self.__create_cipher()
        padded_plaintext = pad(plaintext, self.__block_size)
        ciphertext = cipher.encrypt(padded_plaintext)

        nonce = None
        if 'nonce' in cipher.__dict__.keys():
            nonce = cipher.nonce

        iv = None
        if 'iv' in cipher.__dict__.keys():
            iv = cipher.iv

        # Generate the next key to be used
        if self.__randomize_key_on_every_encryption:
            self.__encryption_key = get_random_bytes(self.__encryption_key_size)

        return ciphertext, nonce, iv

    def decrypt(self, ciphertext: bytes, nonce=None, iv=None):
        """
        Decrypts the ciphertext
        :param ciphertext: The ciphertext to decrypt
        :param nonce: If a nonce was returned for the encryption it is needed here
        :param iv: If an IV was returned for the encryption it is needed here
        :return: the plaintext
        """
        if self.__encryption_key is None:
            raise Exception("Encryption key not set")

        cipher = self.__create_cipher(nonce=nonce, iv=iv)

        padded_plaintext = cipher.decrypt(ciphertext)
        plaintext = unpad(padded_plaintext, self.__block_size)
        return plaintext

    def __create_cipher(self, nonce=None, iv=None):
        """
        Internal method for creation of a pycryptodome cipher
        :param nonce:
        :param iv:
        :return:
        """
        cipher = None
        if self.__encryption_method == EncryptionMethod.AES:
            if nonce is not None:
                cipher = AES.new(self.__encryption_key, _block_mode_dict[self.__block_mode], nonce=nonce)
            elif iv is not None:
                cipher = AES.new(self.__encryption_key, _block_mode_dict[self.__block_mode], iv=iv)
            else:
                cipher = AES.new(self.__encryption_key, _block_mode_dict[self.__block_mode])
        elif self.__encryption_method == EncryptionMethod.DES3:
            if nonce is not None:
                cipher = DES3.new(self.__encryption_key, _block_mode_dict[self.__block_mode], nonce=nonce)
            elif iv is not None:
                cipher = DES3.new(self.__encryption_key, _block_mode_dict[self.__block_mode], iv=iv)
            else:
                cipher = DES3.new(self.__encryption_key, _block_mode_dict[self.__block_mode])
        elif self.__encryption_method == EncryptionMethod.DES:
            if nonce is not None:
                cipher = DES.new(self.__encryption_key, _block_mode_dict[self.__block_mode], nonce=nonce)
            elif iv is not None:
                cipher = DES.new(self.__encryption_key, _block_mode_dict[self.__block_mode], iv=iv)
            else:
                cipher = DES.new(self.__encryption_key, _block_mode_dict[self.__block_mode])
        elif self.__encryption_method == EncryptionMethod.SHIFT:
            if not self.__block_mode == BlockMode.ECB:
                raise Exception("Shift only supports ECB")
            cipher = SimpleShiftCipher(self.__encryption_key)
        elif self.__encryption_method == EncryptionMethod.XOR:
            if not self.__block_mode == BlockMode.ECB:
                raise Exception("XOR only supports ECB")
            cipher = SimpleXorCipher(self.__encryption_key)
        else:
            raise Exception("Unknown encryption method " + str(self.__encryption_method))
        return cipher

    def get_key(self):
        """
        Returns the key for the next encryption operation
        :return: The next key to be used
        """
        return self.__encryption_key
