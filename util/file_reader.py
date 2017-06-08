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

import scipy.io as sio
import random as random
import numpy as np
from scipy import ndimage
from collections import Counter


class FileReader():
    def __init__(self, data_dir, data_info):

        self.data = sio.loadmat(data_dir + data_info)
        self.prefix = data_dir
        self.age = np.squeeze(self.data['celebrityImageData']['age'][0][0])
        self.identity = np.squeeze(self.data['celebrityImageData']['identity'][0][0])
        self.nof_identity = 2000
        self.path = np.squeeze(self.data['celebrityImageData']['name'][0][0]).tolist()
        self.nof_images_at_identity = np.zeros([self.nof_identity, 1])
        for i in self.identity:
            self.nof_images_at_identity[i - 1] += 1

    def __str__(self):
        return 'Data directory:\t' + self.prefix + '\nIdentity Num:\t' + str(self.nof_identity)

    def select_identity(self, nof_person, nof_images):
        images_and_labels = []
        ids_selected = random.sample(xrange(self.nof_identity), nof_person)
        for i in ids_selected:
            images_indices = np.squeeze(np.where(self.identity == i))
            images_selected = random.sample(images_indices, nof_images)
            for image in images_selected:
                images_and_labels.append([image, i])

        image_data = []
        label_data = []
        for image, label in images_and_labels:
            image_data.append(ndimage.imread(self.prefix + self.path[image][0].encode('utf-8')))
            label_data.append(label)
        return image_data, label_data
