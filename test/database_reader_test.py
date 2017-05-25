from unittest import TestCase
from util import database_reader
import scipy.io as sio


class DatabaseReaderTest(TestCase):
    def testDatabaseReader(self):
        batch_size = 10
        dataReader = database_reader.DatabaseReader('/home/bingzhang/Documents/Dataset/CACD/Processed_Aligned', 163446,
                                                    batch_size, 0.8,
                                                    True)
        image, label = dataReader.next_batch(True)
        sio.savemat('batch.mat', {'image': image,
                                  'label': label})
