from unittest import TestCase
from util import file_reader
from util import celebrity
class FileReaderTest(TestCase):

    CACD = file_reader.FileReader('/home/bingzhang/Documents/Dataset/CACD/','cele.mat')
    # file_reader.__str__()
    cele1 = celebrity.Celebrity(CACD.age[0], CACD.identity[0], str(CACD.path[0]))
    print cele1.__str__()
    print CACD.__str__()
    print CACD.select_identity(2,10)