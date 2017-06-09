import unittest
import tensorflow as tf
import numpy as np
class ReshapeTest(unittest.TestCase):
    def test_reshape(self):
        data = tf.placeholder(dtype=tf.int32,shape=[30,20])
        input = np.arange(0,30*20)
        input = np.reshape(input,[30,20])

        b = data[0:30:3][:]
        c = data[1:30:3][:]
        d = data[2:30:3][:]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            a,b1,c1,d1= sess.run([data,b,c,d],{data:input})

        print 'done'

if __name__ == '__main__':
    unittest.main()
