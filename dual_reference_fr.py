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
        self.delta = 0.05  # delta in hinge loss
        self.nof_sampled_id = 20
        self.nof_images_per_id = 20
        self.nof_sampled_age = 20
        self.nof_images_per_age = 20

        """model input placeholder"""
        self.image_in = tf.placeholder(tf.float32, [None, 250, 250, 3], name='image_in')
        self.label_in = tf.placeholder(tf.float32, [None], name='label_in')
        self.val_acc = tf.placeholder(tf.float32, name='val_acc')
        self.dis_check = tf.placeholder(tf.float32, [None, 1, 200, 1])
        """model nodes and ops"""
        self.feature = self.net_forward()
        self.id_embeddings = self.get_id_embeddings(self.feature)
        self.age_embeddings = self.get_age_embeddings(self.feature)
        self.id_loss = self.get_triplet_loss(self.id_embeddings)
        self.age_loss = self.get_triplet_loss(self.age_embeddings)
        self.id_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1).minimize(
            self.id_loss)
        self.age_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1).minimize(
            self.age_loss)

        gpu_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=gpu_config)

    def net_forward(self):
        net, _ = nb.inference(images=self.image_in, keep_probability=1.0, bottleneck_layer_size=128, phase_train=True,
                              weight_decay=0.0, reuse=None)
        feature = slim.fully_connected(net, self.feature_dim, activation_fn=None,
                                       weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       weights_regularizer=slim.l2_regularizer(0.0))
        return feature

    def get_id_embeddings(self, feature):
        with tf.variable_scope('id_embedding'):
            id_embeddings = slim.fully_connected(feature, self.embedding_size, activation_fn=None,
                                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 weights_regularizer=slim.l2_regularizer(0.0), scope='id_embedding')
            weights = slim.get_model_variables('id_embedding')[0]
            bias = slim.get_model_variables('id_embedding')[1]
            nb.variable_summaries(weights, 'weight')
            nb.variable_summaries(bias, 'bias')
            id_embeddings = tf.nn.l2_normalize(id_embeddings, dim=1, epsilon=1e-12, name='id_embeddings')
        return id_embeddings

    def get_age_embeddings(self, feature):
        with tf.variable_scope('age_embedding'):
            age_embeddings = slim.fully_connected(feature, self.embedding_size, activation_fn=None,
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                  weights_regularizer=slim.l2_regularizer(0.0), scope='age_embedding')
            weights = slim.get_model_variables('age_embedding')[0]
            bias = slim.get_model_variables('age_embedding')[1]
            nb.variable_summaries(weights, 'weight')
            nb.variable_summaries(bias, 'bias')
            # tf.summary.scalar('age_embedding',slim.get_model_variables('age_embedding'))
            id_embeddings = tf.nn.l2_normalize(age_embeddings, dim=1, epsilon=1e-12, name='age_embeddings')
        return id_embeddings

    def get_triplet_loss(self, embeddings):
        anchor = embeddings[0:self.batch_size:3][:]
        positive = embeddings[1:self.batch_size:3][:]
        negative = embeddings[2:self.batch_size:3][:]

        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.delta)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        return loss

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        step = 0
        saver = tf.train.Saver()
        writer_train = tf.summary.FileWriter(self.paths.log_dir, self.sess.graph)
        CACD = fr.FileReader(self.paths.data_dir, 'cele.mat', contain_val=True, val_data_dir=self.paths.val_dir,
                             val_list=self.paths.val_list)
        tf.summary.image('input', self.image_in, 10)
        tf.summary.image('dis', self.dis_check, 1)
        tf.summary.scalar('id_loss', self.id_loss)
        dis = np.zeros(200)
        summary_op = tf.summary.merge_all()
        triplet_select_times = 1
        while triplet_select_times < 19999:
            # ID step
            print '!!!!!!!!!ID!!!!!!!!!!start forward propagation on a SAMPLE_BATCH (nof_sampled_id,nof_image_per_id)=(%d,%d)' % (
                self.nof_sampled_id, self.nof_images_per_id)
            time_start = time.time()
            image, label, image_path, sampled_id = CACD.select_identity(self.nof_sampled_id, self.nof_images_per_id)
            id_emb = self.sess.run(self.id_embeddings, feed_dict={self.image_in: image, self.label_in: label})
            print 'Time Elapsed %lf' % (time.time() - time_start)
            time_start = time.time()
            print '[%d]selecting id triplets' % triplet_select_times
            triplet = triplet_sample(id_emb, self.nof_sampled_id, self.nof_images_per_id, self.delta)
            nof_triplet = len(triplet)

            print 'num of selected id triplets:%d' % nof_triplet
            print 'Time Elapsed:%lf' % (time.time() - time_start)
            inner_step = 0
            for i in xrange(0, nof_triplet, self.batch_size // 3):
                if i + self.batch_size // 3 < nof_triplet:
                    triplet_image, triplet_label = CACD.read_triplet(image_path, label, triplet, i,
                                                                     self.batch_size // 3)
                    triplet_image = np.reshape(triplet_image, [-1, 250, 250, 3])
                    triplet_label = np.reshape(triplet_label, [-1])
                    start_time = time.time()
                    err, summary, _ = self.sess.run([self.id_loss, summary_op, self.id_opt],
                                                    feed_dict={self.image_in: triplet_image,
                                                               self.label_in: triplet_label,
                                                               self.dis_check: np.reshape(np.array(dis),
                                                                                          [-1, 1, 200, 1])})
                    print '[%d/%d@%dth select_triplet & global_step %d] \033[1;31;40m loss:[%lf] \033[1;m time elapsed:%lf' % (
                        inner_step, (nof_triplet * 3) // self.batch_size, triplet_select_times, step, err,
                        time.time() - start_time)
                    writer_train.add_summary(summary, step)
                    step += 1
                    inner_step += 1
                    if step % 20 == 0:
                        val_iters = CACD.val_size // 20
                        ground_truth = []
                        emb = []
                        # extract embeddings by batch as GPU memory is not enough
                        for _ in range(val_iters):
                            val_data, val_label = CACD.get_test(20)
                            val_data = np.reshape(val_data, [-1, 250, 250, 3])
                            ground_truth.append(val_label)
                            emb_batch = self.sess.run(self.id_embeddings, feed_dict={self.image_in: val_data})
                            emb.append(emb_batch)
                        ground_truth = np.reshape(ground_truth, (-1,))
                        emb = np.reshape(emb, (-1, 128))
                        for j in range(CACD.val_size):
                            dis[j] = np.sum(np.square(emb[j * 2] - emb[j * 2 + 1]))
                    if step % 10000 == 0:
                        saver.save(self.sess, 'QModel', step)
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
    np.random.shuffle(triplet)
    return triplet


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--workplace', type=str,
                        help='where the code runs', default='server')

    return parser.parse_args(argv)


def get_rank_k(aff, k):
    temp = np.argsort(aff)
    ranks = np.arange(len(aff))[np.argsort(temp)]
    ranks[np.where(ranks > k)] = 255
    return ranks


if __name__ == '__main__':
    config = configurer.Configurer(parse_arguments(sys.argv[1:]).workplace)
    model = DualReferenceFR(config)
    model.train()
