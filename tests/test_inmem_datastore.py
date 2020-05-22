import unittest

from datastore.datastore import InMemDatastore


class InMemDatastoreTester(unittest.TestCase):
    def test_inmem_datastore(self):
        fields = list(map(lambda x: str(x), range(1, 11)))
        datastore = InMemDatastore(fields)

        # Write all
        for i in range(1, 11):
            record = {}
            for j in range(1, 11):
                record[str(j)] = "num=" + str(i * j)

            datastore.write_record(record)

        self.assertEqual(10, datastore.pending_num_of_records())

        # Read all
        i = 1
        while datastore.has_more_records():
            next_record = datastore.read_next_record()
            for key, value in next_record.items():
                if type(value) == bytes or type(value) == bytearray:
                    value = str(value, encoding="UTF-8")

                self.assertEqual("num=" + str(i * int(key)), value)
            i += 1

        self.assertEqual(0, datastore.pending_num_of_records())


if __name__ == '__main__':
    unittest.main()
