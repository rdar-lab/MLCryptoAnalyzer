import os
import tempfile
import unittest

from datastore.datastore import FileDatastore


class FileDatastoreTester(unittest.TestCase):
    def test_file_datastore(self):
        file = tempfile.TemporaryFile()
        file_name = file.name
        file.close()

        fields = list(map(lambda x: str(x), range(1, 11)))
        datastore = FileDatastore(fields, file_name, "wb")

        # Write all
        for i in range(1, 11):
            record = {}
            for j in range(1, 11):
                record[str(j)] = "num=" + str(i * j)

            datastore.write_record(record)
        datastore.close()

        self.assertTrue(os.path.exists(file_name))
        self.assertTrue(os.path.isfile(file_name))
        self.assertTrue(os.path.getsize(file_name) > 0)

        # Read all
        datastore = FileDatastore(fields, file_name, "rb")
        i = 1
        next_record = datastore.read_next_record()
        while next_record is not None:
            for key, value in next_record.items():
                if type(value) == bytes or type(value) == bytearray:
                    value = str(value, encoding="UTF-8")

                self.assertEqual("num=" + str(i * int(key)), value)
            i += 1
            next_record = datastore.read_next_record()
        datastore.close()

        self.assertEqual(11, i)


if __name__ == '__main__':
    unittest.main()
