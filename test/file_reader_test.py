from unittest import TestCase
from util import file_reader
from util import celebrity
import scipy.io as sio
import numpy as np

class FileReaderTest(TestCase):
    CACD = file_reader.FileReader('C:/Users/BingZhang/Documents/MATLAB/CACD/CACD2000/', 'cele.mat', contain_val=False,
                                  val_data_dir='/home/bingzhang/Documents/Dataset/lfw/',
                                  val_list='/home/bingzhang/Documents/Dataset/ZID/LFW/lfw_trip_val.txt')
    # file_reader.__str__()
    # cele1 = celebrity.Celebrity(CACD.age[0], CACD.identity[0], str(CACD.path[0]))
    data = CACD.get_next_batch(2)
    sio.savemat('./test.mat',{'data':data})
    print(CACD.__str__())
    _, label_data, image_path, ages =CACD.select_age(3,20)
    print(ages)
    print(image_path)
    print(label_data)
    # data = CACD.select_identity(2, 2)
    # sio.savemat('test.mat', {'data': data})
    #
    # validate_data, validate_label = CACD.get_test(10)
    # validate_data= np.reshape(validate_data, [-1, 250, 250, 3])
    # sio.savemat('val.mat', {'val': validate_data})
