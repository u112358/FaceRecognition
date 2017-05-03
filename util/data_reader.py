# MIT License
#
# Copyright (c) 2017 BingZhang Hu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import random as random
import scipy.io as sio
import os.path as op


class DataReader():
    """Read data from *.mat file and generate batches for tensorflow.


    Attributes:
        data_dir: [String] Directory of *.mat files.
        batch_number: [Integer] Number of required batches.
        image_number: [Integer] Number of total images (*.mat files).
        batch_size: [Integer] Number of images within a batch
        current_batch_index: [Integer] Index of current batch (called while reading).

    """

    def __init__(self, data_dir, image_number, batch_size, train_test_ratio, reproducible=True):
        """

        """
        if reproducible:
            np.random.seed(112358)
        self.data_dir = data_dir
        self.image_number = image_number
        self.batch_size = batch_size
        self.train_test_ratio = train_test_ratio
        self.reproducible = reproducible
        self.size_train = image_number * train_test_ratio
        self.size_test = image_number - self.size_train
        self.train_batches = int(np.floor(self.size_train / batch_size))
        self.test_batches = int(np.floor(self.size_test / batch_size))
        print(
            'Initializing Data Reader from: %s\nReproducible: %s\nImage number: %d\nBatch size: %d\nTrain batches: %d\nTest batches: %d\n' % (
                self.data_dir, self.reproducible.__str__(), self.image_number, self.batch_size, self.train_batches,
                self.test_batches))

        self.shuffled_index = range(self.image_number)
        print self.shuffled_index
        np.random.shuffle(self.shuffled_index)
        print self.shuffled_index
        # if image_number % batch_number:
        #     raise Exception('${image_number}[%d]/${batch_number}[%d] must be a Integer.' % (image_number, batch_number))
        # else:
        #     self.data_dir = data_dir
        #     self.batch_number = batch_number
        #     self.image_number = image_number
        #     self.batch_size = image_number / batch_number
        #     print('Initializing Data Reader...\n#Images:\t%d\n#Batches:\t%d\n#Batch size:\t%d'
        #           % (self.image_number, self.batch_number, self.batch_size))
        #
        #     self.image_index = np.arange(image_number) + 1
        #     np.random.shuffle(self.image_index)
        #
        #     self.shuffled_batch = self.image_index.reshape(self.batch_number, self.batch_size)
        #     self.current_batch_index = 0
        #     self.epoch = 1;

    def next_batch(self):
        image_data = []
        label_data = []
        if self.current_batch_index >= self.batch_number:
            np.random.shuffle(self.image_index)
            self.shuffled_batch = self.image_index.reshape(self.batch_number, self.batch_size)
            self.current_batch_index = 0
            string = op.join('------------ finished ', str(self.epoch))
            string = op.join(string, ' epoch---------------')
            print(string)
            self.epoch += 1
        for i in range(self.batch_size):
            file_name = self.data_dir + '/' + str(self.shuffled_batch[self.current_batch_index, i]) + '.mat'
            current_mat = sio.loadmat(file_name)
            image_data = np.append(image_data, current_mat['im'])
            label_data = np.append(label_data, current_mat['label'])
        image_data = image_data.reshape([self.batch_size, image_data.size / self.batch_size])
        label_data = label_data.reshape([self.batch_size, label_data.size / self.batch_size])
        self.current_batch_index += 1
        return image_data, label_data

    def get_test(self, size):
        image_data = []
        label_data = []
        if size > self.image_number:
            raise Exception('input ${size}[%d] larger than test ${image_image}[%d]' % (size, self.image_number))
        else:
            np.random.shuffle(self.image_index)
            for i in range(size):
                file_name = self.data_dir + "/" + str(self.image_index[i]) + ".mat"
                current_mat = sio.loadmat(file_name)
                image_data = np.append(image_data, current_mat['im'])
                label_data = np.append(label_data, current_mat['label'])
            image_data = image_data.reshape(size, image_data.size / size)
            label_data = label_data.reshape(size, label_data.size / size)
            self.current_batch_index += 1
            return image_data, label_data
