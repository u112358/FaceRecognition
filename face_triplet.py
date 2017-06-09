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
from util import data_reader as dr
from util import file_reader as fr
import scipy.io as sio
import tensorflow.contrib.slim as slim
import argparse
import sys
import configurer


class FaceTriplet():
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self, sess, config):
        self.sess = sess
        self.root_dir = os.getcwd()
        self.subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(os.path.expanduser('logs'), self.subdir)
        self.model_dir = os.path.join(os.path.expanduser('models'), self.subdir)
        self.learning_rate = 0.1
        self.batch_size = 30
        self.embedding_size = 2000
        self.max_epoch = 20
        self.data_dir = config.data_dir
        self.model = config.model
        self.image_in = tf.placeholder(tf.float32, [None, 250, 250, 3])
        self.label_in = tf.placeholder(tf.float32, [None])
        self.net = self._build_net()
        self.embeddings = self._forward()
        self.loss = self._build_loss()
        self.accuracy = self._build_accuracy()
        self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1).minimize(self.loss)
        self.nof_sampled_id = 20
        self.nof_images_per_id = 20
        self.delta = 0.5

    def _forward(self):
        net, _ = nb.inference(images=self.image_in, keep_probability=1.0, bottleneck_layer_size=128, phase_train=True,
                              weight_decay=0.0, reuse=True)
        logits = slim.fully_connected(net, self.embedding_size, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(0.0), scope='logits', reuse=True)
        embeddings = tf.nn.l2_normalize(logits, dim=1, epsilon=1e-12, name='embeddings')
        return embeddings

    def _build_net(self):
        # convolution layers
        net, _ = nb.inference(images=self.image_in, keep_probability=1.0, bottleneck_layer_size=128, phase_train=True,
                              weight_decay=0.0)

        # with tf.variable_scope('output') as scope:
        #     weights = tf.get_variable('weights', [1024, self.class_num], dtype=tf.float32,
        #                               initializer=tf.truncated_normal_initializer(stddev=1e-2))
        #     biases = tf.get_variable('biases', [self.class_num], dtype=tf.float32, initializer=tf.constant_initializer())
        #     output = tf.add(tf.matmul(net, weights), biases, name=scope.name)
        #     nb.variable_summaries(weights,'weights')
        #     nb.variable_summaries(biases,'biases')
        logits = slim.fully_connected(net, self.embedding_size, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(0.0), scope='logits')
        return logits

    def _build_loss(self):
        embeddings = self._forward()
        embeddings = tf.reshape(embeddings, [3, 2000, -1])
        anchor = embeddings[0][:][:]
        pos = embeddings[1][:][:]
        neg = embeddings[2][:][:]

        total_loss = triplet_loss(anchor=anchor, positive=pos, negative=neg, delta=0.5)
        return total_loss

    def _build_accuracy(self):
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.net, 1), tf.cast(self.label_in, tf.int64))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def train(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model)
        # saver = tf.train.Saver()
        CACD = fr.FileReader(self.data_dir, 'cele.mat')
        triplet_select_times = 1
        tf.summary.image('image', self.image_in, 10)
        summary_op = tf.summary.merge_all()
        writer_train = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        writer_test = tf.summary.FileWriter(self.log_dir + '/test', self.sess.graph)
        step = 1

        while triplet_select_times < 19999:
            print 'start forward propagation on a SAMPLE_BATCH (nof_sampled_id,nof_image_per_id)=(%d,%d)' % (
                self.nof_sampled_id, self.nof_images_per_id)
            time_start = time.time()
            image, label = CACD.select_identity(self.nof_sampled_id, self.nof_images_per_id)
            emb = self.sess.run(self.embeddings, feed_dict={self.image_in: image, self.label_in: label})
            sio.savemat('emb.mat', {'emb': emb})
            print 'Time Elapsed %lf' % (time.time() - time_start)

            time_start = time.time()
            print '[%d]selecting triplets' % triplet_select_times
            triplet = triplet_sample(emb, self.nof_sampled_id, self.nof_images_per_id, self.delta)
            nof_triplet = len(triplet)
            print 'num of selected triplets:%d' % nof_triplet
            print 'Time Elapsed:%lf' % (time.time() - time_start)
            inner_step = 0
            for i in xrange(0, nof_triplet, self.batch_size // 3):
                if i + self.batch_size // 3 < nof_triplet:
                    triplet_image, triplet_label = CACD.read_triplet(triplet, i, self.batch_size // 3)
                    triplet_image = np.reshape(triplet_image, [-1, 250, 250, 3])
                    triplet_label = np.reshape(triplet_label, [-1])
                    start_time = time.time()
                    err, _ = self.sess.run([self.loss, self.opt],
                                           feed_dict={self.image_in: triplet_image, self.label_in: triplet_label})
                    print '[%d/%d@%dth select_triplet & global_step %d] loss:[%lf] time elapsed:%lf' % (
                        inner_step, (nof_triplet * 3) // self.batch_size, triplet_select_times, step, err,
                        time.time() - start_time)
                    step += 1
                    inner_step += 1
            triplet_select_times += 1


def triplet_sample(embeddings, nof_ids, nof_images_per_id, delta):
    aff = []
    triplet = []
    for anchor_id in xrange(nof_ids * nof_images_per_id):
        dist = np.sum(np.square(embeddings - embeddings[anchor_id]), 1)
        aff.append(dist)
        for pos_id in xrange(anchor_id + 1, (anchor_id // nof_images_per_id + 1) * nof_images_per_id):
            neg_dist = np.copy(dist)
            neg_dist[anchor_id:(anchor_id // nof_images_per_id + 1) * nof_images_per_id] = np.NAN
            neg_ids = np.where(neg_dist - dist[pos_id] < delta)[0]
            nof_neg_ids = len(neg_ids)
            if nof_neg_ids > 0:
                rand_id = np.random.randint(nof_neg_ids)
                neg_id = neg_ids[rand_id]
                triplet.append([anchor_id, pos_id, neg_id])
    return triplet


def triplet_loss(anchor, positive, negative, delta):
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 0)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 0)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), delta)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--workplace', type=str,
                        help='where the code runs', default='server')

    return parser.parse_args(argv)


if __name__ == '__main__':
    this_session = tf.Session()
    config = configurer.Configurer(parse_arguments(sys.argv[1:]).workplace)
    model = FaceTriplet(this_session, config)
    model.train()
