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


class FaceQuartet():
    def __init__(self, sess, config):
        self.sess = sess
        self.subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(os.path.expanduser('logs/quartet'), self.subdir)
        self.data_dir = config.data_dir
        self.model = config.model
        self.val_dir = config.val_dir
        self.val_list = config.val_list
        self.learning_rate = 0.008
        self.batch_size = 30
        self.embedding_size = 2000
        self.max_epoch = 20
        self.delta = 0.25
        self.nof_sampled_id = 20
        self.nof_images_per_id = 20
        self.nof_sampled_age = 20
        self.nof_images_per_age = 20
        self.image_in = tf.placeholder(tf.float32, [None, 250, 250, 3])
        self.label_in = tf.placeholder(tf.float32, [None])
        # confusion matrix to display in tensorboard
        self.affinity = tf.placeholder(tf.float32, [None, self.nof_images_per_id * self.nof_sampled_id,
                                                    self.nof_images_per_id * self.nof_sampled_id, 1])
        # binariesed confusion matrix
        self.result = tf.placeholder(tf.float32, [None, self.nof_images_per_id * self.nof_sampled_id,
                                                  self.nof_images_per_id * self.nof_sampled_id, 1])
        # a pattern to monitor the identity sampling
        self.possible_triplets = tf.placeholder(tf.int16, name='nof_possible_triplets')
        self.val_acc = tf.placeholder(tf.float32, name='val_accuracy')
        self.sampled_freq = tf.placeholder(tf.float32, [1, 50, 40, 1], name='sampled_freq_image')
        self.id_embeddings, self.age_embeddings = self._forward()
        self.id_loss = self._build_loss(self.id_embeddings,'id_loss')
        self.age_loss = self._build_loss(self.age_embeddings,'age_loss')
        # self.loss = self.id_loss+self.age_loss
        self.id_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1).minimize(self.id_loss)
        self.age_opt = tf.train.AdamOptimizer(self.learning_rate,beta1=0.9,beta2=0.999,epsilon=0.1).minimize(self.age_loss)

    def _forward(self):
        net, _ = nb.inference(images=self.image_in, keep_probability=1.0, bottleneck_layer_size=128, phase_train=True,
                              weight_decay=0.0, reuse=None)
        logits = slim.fully_connected(net, self.embedding_size, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(0.0), scope='logits')
        id_logits = slim.fully_connected(logits,128,activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                         weights_regularizer=slim.l2_regularizer(0.0), scope='id_logits')
        age_logits = slim.fully_connected(logits,128,activation_fn=None,
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                         weights_regularizer=slim.l2_regularizer(0.0), scope='age_logits')
        id_embeddings = tf.nn.l2_normalize(id_logits, dim=1, epsilon=1e-12, name='id_embeddings')
        age_embeddings = tf.nn.l2_normalize(age_logits,dim=1,epsilon=1e-12,name='age_embeddings')
        return id_embeddings, age_embeddings

    def _build_loss(self,embeddings,name):
        anchor = embeddings[0:self.batch_size:3][:]
        pos = embeddings[1:self.batch_size:3][:]
        neg = embeddings[2:self.batch_size:3][:]

        total_loss = triplet_loss(anchor=anchor, positive=pos, negative=neg, delta=self.delta)

        tf.summary.scalar(name, total_loss)
        return total_loss

    # def _update_lr(self,lr):
    #     self.learning_rate = lr
    #     self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1).minimize(self.loss)
    def train(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        step = 0
        saver = tf.train.Saver()
        # saver.restore(self.sess, self.model)
        # step = 120000
        # saver = tf.train.Saver()
        CACD = fr.FileReader(self.data_dir, 'cele.mat', contain_val=True, val_data_dir=self.val_dir,
                             val_list=self.val_list)
        triplet_select_times = 1
        writer_train = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        # writer_train_1 = tf.summary.FileWriter(self.log_dir + '/train/margin=0.5', self.sess.graph)
        # writer_train_2 = tf.summary.FileWriter(self.log_dir + '/train/margin=1', self.sess.graph)
        # writer_train_3 = tf.summary.FileWriter(self.log_dir + '/train/margin=1.3', self.sess.graph)
        # writer_train_4 = tf.summary.FileWriter(self.log_dir + '/train/margin=1.6', self.sess.graph)
        sampled_freq = np.zeros([2000, 1])
        acc = 0
        with tf.name_scope('ToCheck'):
            tf.summary.image('affinity', self.affinity, 1)
            tf.summary.image('result', self.result)
        tf.summary.scalar('possible triplets', self.possible_triplets)
        tf.summary.image('sampled_freq', self.sampled_freq)

        summary_op = tf.summary.merge_all()
        val_summary_op = tf.summary.scalar('val_acc', self.val_acc)
        while triplet_select_times < 19999:
            # ID step
            print '!!!!!!!!!ID!!!!!!!!!!start forward propagation on a SAMPLE_BATCH (nof_sampled_id,nof_image_per_id)=(%d,%d)' % (
                self.nof_sampled_id, self.nof_images_per_id)
            time_start = time.time()
            image, label, image_path, sampled_id = CACD.select_identity(self.nof_sampled_id, self.nof_images_per_id)
            sampled_freq[sampled_id] += 1
            id_emb = self.sess.run(self.id_embeddings, feed_dict={self.image_in: image, self.label_in: label})
            aff = []
            for idx in range(len(label)):
                aff.append(np.sum(np.square(id_emb[idx][:] - id_emb), 1))
            result = get_rank_k(aff, self.nof_images_per_id)

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
                                                               self.affinity: np.reshape(np.array(aff), [-1,
                                                                                                         self.nof_images_per_id * self.nof_sampled_id,
                                                                                                         self.nof_images_per_id * self.nof_sampled_id,
                                                                                                         1]),
                                                               self.possible_triplets: nof_triplet,
                                                               self.val_acc: acc,
                                                               self.result: np.reshape(result, [-1,
                                                                                                self.nof_images_per_id * self.nof_sampled_id,
                                                                                                self.nof_images_per_id * self.nof_sampled_id,
                                                                                                1]),
                                                               self.sampled_freq: np.reshape(sampled_freq,
                                                                                             [1, 50, 40, 1])})
                    print '[%d/%d@%dth select_triplet & global_step %d] \033[1;31;40m loss:[%lf] \033[1;m time elapsed:%lf' % (
                        inner_step, (nof_triplet * 3) // self.batch_size, triplet_select_times, step, err,
                        time.time() - start_time)
                    writer_train.add_summary(summary, step)
                    step += 1
                    inner_step += 1
                    if inner_step % 5 == 0:
                        emb = self.sess.run(self.id_embeddings, feed_dict={self.image_in: image, self.label_in: label})
                        aff = []
                        for idx in range(len(label)):
                            aff.append(np.sum(np.square(emb[idx][:] - emb), 1))
                        result = get_rank_k(aff, self.nof_images_per_id)
                    # if step % 200 == 0:
                    #     # perform validate
                    #     val_iters = CACD.val_size // 20
                    #     true_label = []
                    #     emb = []
                    #     for _ in range(val_iters):
                    #         validate_data, validate_label = CACD.get_test(20)
                    #         validate_data = np.reshape(validate_data, [-1, 250, 250, 3])
                    #         true_label.append(validate_label)
                    #         emb_bacth = self.sess.run(self.embeddings, feed_dict={self.image_in: validate_data})
                    #         emb.append(emb_bacth)
                    #     true_label = np.reshape(true_label, (-1,))
                    #     emb = np.reshape(emb, (-1, self.embedding_size))
                    #     pre_label = []
                    #     for j in range(CACD.val_size):
                    #         if np.sum(np.square(emb[j * 2] - emb[j * 2 + 1])) < 0.5:
                    #             pre_label.append(1)
                    #         else:
                    #             pre_label.append(0)
                    #     correct = np.sum(abs(np.array(pre_label) - np.array(true_label)))
                    #     acc = float(correct) / CACD.val_size
                    #     sum = self.sess.run(val_summary_op, feed_dict={self.val_acc: acc})
                    #     writer_train_1.add_summary(sum, step)
                    #
                    #     pre_label = []
                    #     for j in range(CACD.val_size):
                    #         if np.sum(np.square(emb[j * 2] - emb[j * 2 + 1])) < 1:
                    #             pre_label.append(1)
                    #         else:
                    #             pre_label.append(0)
                    #     correct = np.sum(abs(np.array(pre_label) - np.array(true_label)))
                    #     acc = float(correct) / CACD.val_size
                    #     sum = self.sess.run(val_summary_op, feed_dict={self.val_acc: acc})
                    #     writer_train_2.add_summary(sum, step)
                    #
                    #     pre_label = []
                    #     for j in range(CACD.val_size):
                    #         if np.sum(np.square(emb[j * 2] - emb[j * 2 + 1])) < 1.3:
                    #             pre_label.append(1)
                    #         else:
                    #             pre_label.append(0)
                    #     correct = np.sum(abs(np.array(pre_label) - np.array(true_label)))
                    #     acc = float(correct) / CACD.val_size
                    #     sum = self.sess.run(val_summary_op, feed_dict={self.val_acc: acc})
                    #     writer_train_3.add_summary(sum, step)
                    #
                    #     pre_label = []
                    #     for j in range(CACD.val_size):
                    #         if np.sum(np.square(emb[j * 2] - emb[j * 2 + 1])) < 1.6:
                    #             pre_label.append(1)
                    #         else:
                    #             pre_label.append(0)
                    #     correct = np.sum(abs(np.array(pre_label) - np.array(true_label)))
                    #     acc = float(correct) / CACD.val_size
                    #     sum = self.sess.run(val_summary_op, feed_dict={self.val_acc: acc})
                    #     writer_train_4.add_summary(sum, step)
                    if step %10000 ==0:
                        saver.save(self.sess,'QModel',step)

            # AGE step
            print '!!!!!!!!!AGE!!!!!!!!!!start forward propagation on a SAMPLE_BATCH (nof_sampled_age,nof_image_per_age)=(%d,%d)' % (
                self.nof_sampled_age, self.nof_images_per_age)
            time_start = time.time()
            image, label, image_path, sampled_id = CACD.select_age(self.nof_sampled_age,
                                                                        self.nof_images_per_age)

            age_emb = self.sess.run(self.age_embeddings,
                                   feed_dict={self.image_in: image, self.label_in: label})

            print 'Time Elapsed %lf' % (time.time() - time_start)

            time_start = time.time()
            print '[%d]selecting id triplets' % triplet_select_times
            triplet = triplet_sample(age_emb, self.nof_sampled_age, self.nof_images_per_age, self.delta)
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
                    err, summary, _ = self.sess.run([self.age_loss, summary_op, self.age_opt],
                                                    feed_dict={self.image_in: triplet_image,
                                                               self.label_in: triplet_label,
                                                               })
                    print '[%d/%d@%dth select_triplet & global_step %d] \033[1;31;40m loss:[%lf] \033[1;m time elapsed:%lf' % (
                        inner_step, (nof_triplet * 3) // self.batch_size, triplet_select_times, step, err,
                        time.time() - start_time)
                    writer_train.add_summary(summary, step)
                    step += 1
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
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), delta)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss


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
    if not parse_arguments(sys.argv[1:]).workplace == 'sweet_home':
        gpu_config = tf.ConfigProto(allow_soft_placement=True)
        this_session = tf.Session(config=gpu_config)
        model = FaceQuartet(this_session, config)
    else:
        this_session = tf.Session()
        model = FaceQuartet(this_session, config)
    model.train()
