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
import tensorflow as tf


class FileReader():
    def __init__(self, data_dir, data_info, contain_val=False, val_data_dir='', val_list=''):

        self.data = sio.loadmat(data_dir + data_info)
        self.prefix = data_dir
        self.age = np.squeeze(self.data['celebrityImageData']['age'][0][0])
        self.identity = np.squeeze(self.data['celebrityImageData']['identity'][0][0])
        self.nof_identity = 2000
        self.path = np.squeeze(self.data['celebrityImageData']['name'][0][0]).tolist()
        self.nof_images_at_identity = np.zeros([self.nof_identity, 1])
        for i in self.identity:
            self.nof_images_at_identity[i - 1] += 1
        self.val_data_dir = val_data_dir
        self.val_img_pair = []
        self.val_label = []
        self.val_size = 0
        self.current_val_idx = 0
        if contain_val:
            val_file = open(val_list)
            for i in val_file.readlines():
                im1, im2, label = i.split(' ')
                self.val_img_pair.append([self.val_data_dir + im1, self.val_data_dir + im2])
                self.val_label.append(int(label))
                self.val_size += 1
        self.val_size = 200

    def __str__(self):
        return 'Data directory:\t' + self.prefix + '\nIdentity Num:\t' + str(self.nof_identity)

    def select_identity(self, nof_person, nof_images):
        images_and_labels = []
        # ids_selected \in [0,1999]
        ids_selected = random.sample(xrange(self.nof_identity), nof_person)
        for i in ids_selected:
            # here we select id with 'i+1' as the cele.mat stores the identity label from idx 1
            images_indices = np.where(self.identity == i + 1)[0]
            print 'id:%d len:%d' % (i + 1, len(images_indices))
            images_selected = random.sample(images_indices, nof_images)
            for image in images_selected:
                images_and_labels.append([image, i])
        image_data = []
        label_data = []
        image_path = []
        for image, label in images_and_labels:
            image_data.append(self.read_jpeg_image(self.prefix + self.path[image][0].encode('utf-8')))
            label_data.append(label)
            image_path.append(self.prefix + self.path[image][0].encode('utf-8'))
        return image_data, label_data, image_path, ids_selected

    def select_quartet(self,nof_person, nof_images):
        images_and_labels = []
        ages = []
        ids_selected = random.sample(xrange(self.nof_identity), nof_person)
        for i in ids_selected:
            ages.append(self.age[np.where(self.identity==i+1)[0]])
        return ages


    def read_triplet(self, image_path, label, triplet, i, len):
        triplet_image = []
        triplet_label = []
        for idx in xrange(i, i + len):
            anchor = self.read_jpeg_image(image_path[triplet[idx][0]])
            pos = self.read_jpeg_image(image_path[triplet[idx][1]])
            neg = self.read_jpeg_image(image_path[triplet[idx][2]])
            triplet_image.append([anchor, pos, neg])
            triplet_label.append([label[triplet[idx][0]], label[triplet[idx][1]], label[triplet[idx][2]]])
        return triplet_image, triplet_label






    def get_test(self, n):
        data = []
        label = []
        if self.current_val_idx + n > self.val_size:
            self.current_val_idx = 0
        for i in xrange(self.current_val_idx, self.current_val_idx + n):
            image1 = self.read_jpeg_image(self.val_img_pair[i][0])
            image2 = self.read_jpeg_image(self.val_img_pair[i][1])
            data.append([image1, image2])
            label.append(self.val_label[i])
        self.current_val_idx += n
        return data, label

    def read_jpeg_image(self, path):
        content = ndimage.imread(path)
        mean_v = np.mean(content)
        adjustied_std = np.maximum(np.std(content), 1.0 / np.sqrt(250 * 250 * 3))
        content = (content - mean_v) / adjustied_std
        return content
