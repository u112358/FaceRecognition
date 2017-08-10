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

import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import time
from util import net_builder as nb
from util import file_reader as fr
import tensorflow.contrib.slim as slim
import argparse
import sys
import configurer

class DualReferenceFR():

    def __init__(self, env_config):
        """env_config stores all the PATHs used in this script"""
        self.paths = env_config

        """training parameters"""
        self.learning_rate = 0.008
        self.batch_size = 30
        self.feature_dim = 2000
        self.embedding_size = 128
        self.max_epoch = 20
        self.delta = 0.25  # delta in hinge loss
        self.nof_sampled_id = 20
        self.nof_images_per_id  = 20
        self.nof_sampled_age = 20
        self.nof_images_per_age = 20

        """model input placeholder"""
        self.image_in = tf.placeholder(tf.float32,[None,250,250,3],name='image_in')
        self.label_in = tf.placeholder(tf.float32,[None],name='label_in')
        self.val_acc = tf.placeholder(tf.float32,name='val_acc')
        self.feature = self._net_forward()
        self.id_embeddings = self._get_id

        gpu_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=gpu_config)

    def _net_forward(self):
        net, _ = nb.inference(images=self.image_in, keep_probability=1.0, bottleneck_layer_size=128, phase_train=True,
                              weight_decay=0.0, reuse=None)
        feature = slim.fully_connected(net, self.feature_dim, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(0.0), scope='logits')
        return feature

    def _get_id_embeddings(self,feature):
        with tf.variable_scope('id_embedding'):
            id_embeddings = slim.fully_connected(feature,self.embedding_size,activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                             weights_regularizer=slim.l2_regularizer(0.0), scope='id_logits')
            id_embeddings = tf.nn.l2_normalize(id_embeddings, dim=1, epsilon=1e-12, name='id_embeddings')
        return id_embeddings

    def _get_age_embeddings(self,feature):
        with tf.variable_scope('age_embedding'):
            age_embeddings = slim.fully_connected(feature,self.embedding_size,activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                             weights_regularizer=slim.l2_regularizer(0.0), scope='id_logits')
            id_embeddings = tf.nn.l2_normalize(age_embeddings, dim=1, epsilon=1e-12, name='age_embeddings')
        return id_embeddings