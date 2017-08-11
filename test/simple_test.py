import unittest
import tensorflow as tf

class MyTestCase(unittest.TestCase):
    def test_something(self):
        adder = tf.Variable([1.0],tf.float32)
        b = tf.Variable([2.0],tf.float32)
        b = tf.add(adder,b)
        d = [b,tf.add(b,b)]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print sess.run([b,d])
            print sess.run([b,d])
            print sess.run([d,b])


if __name__ == '__main__':
    unittest.main()
