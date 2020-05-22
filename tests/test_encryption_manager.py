import unittest

from Crypto.Random import get_random_bytes

from encryption.encryption_manager import EncryptionManager, EncryptionMethod, BlockMode


class EncryptionManagerTester(unittest.TestCase):

    def _test_specific_enc_specific_mode(self, encryption_mode, key_size, block_mode):
        print("Testing " + encryption_mode + " - " + block_mode + ": ", end="")

        for i in range(10):
            print(".", end="")

            # Short message
            self._test_encrypt_decrypt(encryption_mode, key_size, block_mode, get_random_bytes(8))

            # Long message
            self._test_encrypt_decrypt(encryption_mode, key_size, block_mode, get_random_bytes(1024 * 1024))
        print("")

    def _test_encrypt_decrypt(self, encryption_mode, key_size, block_mode, plaintext):
        encryption_manager = EncryptionManager(encryption_mode, encryption_key_size=key_size, block_mode=block_mode)
        key = encryption_manager.get_key()
        self.assertIsNotNone(key)
        self.assertEqual(bytes, type(key))

        ciphertext, nonce, iv = encryption_manager.encrypt(plaintext)
        self.assertNotEqual(plaintext, ciphertext)

        decrypted = EncryptionManager(encryption_mode, encryption_key=key, block_mode=block_mode).decrypt(
            ciphertext, nonce=nonce, iv=iv)

        self.assertEqual(plaintext, decrypted)

    def test_all(self):
        self._test_specific_enc_specific_mode(EncryptionMethod.AES, 32, BlockMode.ECB)
        self._test_specific_enc_specific_mode(EncryptionMethod.AES, 32, BlockMode.CBC)
        self._test_specific_enc_specific_mode(EncryptionMethod.AES, 32, BlockMode.CFB)
        self._test_specific_enc_specific_mode(EncryptionMethod.AES, 32, BlockMode.OFB)
        self._test_specific_enc_specific_mode(EncryptionMethod.AES, 32, BlockMode.CTR)

        self._test_specific_enc_specific_mode(EncryptionMethod.DES3, 24, BlockMode.ECB)
        self._test_specific_enc_specific_mode(EncryptionMethod.DES3, 24, BlockMode.CBC)
        self._test_specific_enc_specific_mode(EncryptionMethod.DES3, 24, BlockMode.CFB)
        self._test_specific_enc_specific_mode(EncryptionMethod.DES3, 24, BlockMode.OFB)

        self._test_specific_enc_specific_mode(EncryptionMethod.DES, 8, BlockMode.ECB)
        self._test_specific_enc_specific_mode(EncryptionMethod.DES, 8, BlockMode.CBC)
        self._test_specific_enc_specific_mode(EncryptionMethod.DES, 8, BlockMode.CFB)
        self._test_specific_enc_specific_mode(EncryptionMethod.DES, 8, BlockMode.OFB)

        self._test_specific_enc_specific_mode(EncryptionMethod.SHIFT, 1000, BlockMode.ECB)
        self._test_specific_enc_specific_mode(EncryptionMethod.XOR, 1000, BlockMode.ECB)


if __name__ == '__main__':
    unittest.main()
