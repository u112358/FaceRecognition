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

from datetime import datetime
import os
class Configurer():
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """
    def __init__(self, workplace):
        """Inits SampleClass with blah."""
        subdir = datetime.strftime(datetime.now(),'%Y%m%d-%H%M%S')
        self.log_dir = os.path.join('./log',subdir)
        if workplace =='lab':
            self.data_dir = '/home/bingzhang/Documents/Dataset/CACD/CACD2000/'
            self.model = '/home/bingzhang/Workspace/PycharmProjects/model/20170529-141612-52288'
            self.val_dir = '/home/bingzhang/Documents/Dataset/lfw/'
            self.val_list = '/home/bingzhang/Documents/Dataset/ZID/LFW/lfw_val_list.txt'
        elif workplace=='server':
            self.data_dir = '/scratch/BingZhang/dataset/CACD2000/'
            #self.model= '/scratch/BingZhang/FaceRecognition.close/models/20170529-141612-52288'
            self.model = '/scratch/BingZhang/FaceRecognition/DRFRModel-90000'
            self.val_dir = '/scratch/BingZhang/lfw/'
            self.val_list = '/scratch/BingZhang/val.list'
        elif workplace=='sweet_home':
            self.data_dir = '/Users/bingzhang/Documents/Dataset/CACD2000/'
            self.model ='/Users/bingzhang/Documents/Dataset/model/20170529-141612-52288'
            self.val_dir = '/home/bingzhang/Documents/Dataset/lfw/'
            self.val_list = '/home/bingzhang/Documents/Dataset/ZID/LFW/lfw_trip_val.txt'
        else:
            self.data_dir = '/scratch/BingZhang/dataset/CACD2000/'
            self.model = '/scratch/BingZhang/FaceRecognition.close/models/20170529-141612-52288'
            self.val_dir = '/scratch/BingZhang/lfw/'
            self.val_list = '/scratch/BingZhang/val.list'



