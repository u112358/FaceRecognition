from unittest import TestCase
from util import file_reader
from util import celebrity
import scipy.io as sio
import numpy as np


class FileReaderTest(TestCase):
    CACD = file_reader.FileReader('/home/bingzhang/Documents/Dataset/CACD/CACD2000/', 'cele.mat', contain_val=True,
                                  val_data_dir='/home/bingzhang/Documents/Dataset/lfw/',
                                  val_list='/home/bingzhang/Documents/Dataset/ZID/LFW/lfw_trip_val.txt')
    # file_reader.__str__()
    cele1 = celebrity.Celebrity(CACD.age[0], CACD.identity[0], str(CACD.path[0]))
    print cele1.__str__()
    print CACD.__str__()
    data = CACD.select_identity(2, 2)
    sio.savemat('test.mat', {'data': data})

    validate_data, validate_label = CACD.get_test(10)
    validate_data= np.reshape(validate_data, [-1, 250, 250, 3])
    sio.savemat('val.mat', {'val': validate_data})
