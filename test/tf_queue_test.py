from unittest import TestCase
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops

class TfQueueTest(TestCase):
    def test_tf_queue(self):
        tf.InteractiveSession()

        q = tf.FIFOQueue(2, "float")
        init = q.enqueue_many(([0, 0],))

        x = q.dequeue()
        y = x + 1
        q_inc = q.enqueue([y])

        init.run()
        print(q_inc.run())
        print(q_inc.run())
        print(q_inc.run())
        print(q_inc.run())
        print(q_inc.run())
        print x.eval()
        print x.eval()

        # declare a queue node
        index_queue = data_flow_ops.FIFOQueue(capacity=1000,dtypes=[tf.int64],shapes=[()])
        # declare a data node
        data = tf.Variable([1,2,3,3,2,1,1,2,3,4],dtype=tf.int64)
        print tf.global_variables_initializer().run()
        # declare a enqueue op
        index_queue_enqueue = index_queue.enqueue_many(data)

        print index_queue_enqueue.run()

        index_queue_dequeue = index_queue.dequeue_many(5)

        print index_queue_dequeue.eval()

