from unittest import TestCase
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops

class TfQueueTest(TestCase):
    def test_tf_queue(self):
        # tf.InteractiveSession()

        # declare a queue node
        # index_queue = tf.FIFOQueue(capacity=1000,dtypes=[tf.float32],shapes=[()])
        # # declare a variable and initialize
        # data = tf.get_variable(name='data',shape=[(10)],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1e-1))
        # tf.global_variables_initializer().run()
        # # declare a enqueue op
        # index_queue_enqueue = index_queue.enqueue_many(data)
        #
        # index_queue_enqueue.run()
        #
        # index_queue_dequeue = index_queue.dequeue_many(5)
        #
        # print 'Data generated:', data.eval()
        # print index_queue_dequeue.eval()

        # declare


        # index_queue = tf.train.range_input_producer(100, num_epochs=None,
        #                                             shuffle=True, seed=None, capacity=20)
        # index_dequeue_op = index_queue.dequeue_many(5, 'index_dequeue')
        # with tf.Session() as sess:
        #     coord = tf.train.Coordinator()
        #     tf.train.start_queue_runners(coord=coord, sess=sess)
        #     print sess.run(index_dequeue_op)


        labels_placeholder = tf.placeholder(tf.int64, shape=(None,3), name='labels')


        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name='image_paths')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(3,), (3,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])
        dequeue_op = input_queue.dequeue_many(1)

        nrof_preprocess_threads = 4
        dic = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            dic.append([filenames, label])
        image_paths_array=[['path1','path2','path3'],['path4','path5','path6']]
        labels_array=[[1,2,3],[4,5,6]]
        with tf.Session() as sess:
            sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
            print sess.run(dequeue_op)