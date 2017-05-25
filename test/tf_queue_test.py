from unittest import TestCase
import tensorflow as tf

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
