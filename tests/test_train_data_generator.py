import unittest

from datastore.datastore import InMemDatastore
from datastore.encryption_datastore_constants import EncryptionDatastoreConstants
from ml.training_data_generator import TrainingDataGenerator


class TrainDataGeneratorTester(unittest.TestCase):
    def test_train_data_generator(self):
        output = InMemDatastore(EncryptionDatastoreConstants.ALL_FIELDS_DEF)
        generator = TrainingDataGenerator(output, 1000)
        generator.generate_data()

        self.assertEqual(1000, output.pending_num_of_records())


if __name__ == '__main__':
    unittest.main()
