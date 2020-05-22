import unittest

from generators.plaintext_generator import DictionaryPlaintextGenerator
from utils.dict_utils import ENGLISH_DICT_PATH


class DictionaryPlaintextGeneratorTester(unittest.TestCase):
    def test_generate_data(self):
        generator = DictionaryPlaintextGenerator(ENGLISH_DICT_PATH, 100, 1000)
        data = generator.generate()
        self.assertEqual(bytes, type(data))
        self.assertGreaterEqual(len(data), 100)
        self.assertLessEqual(len(data), 2000)  # add some due to text generation may take more space

        sum_of_bytes = 0
        for i in range(0, len(data)):
            self.assertGreaterEqual(data[i], 0)
            self.assertLessEqual(data[i], 255)
            sum_of_bytes = sum_of_bytes + data[i]

        self.assertNotEqual(0, sum_of_bytes / len(data))


if __name__ == '__main__':
    unittest.main()
