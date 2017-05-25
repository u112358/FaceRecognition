import tensorflow as tf
import scipy.io as sio
file = tf.read_file('/home/bingzhang/Documents/Dataset/CACD/Processed_Aligned/1/53_Robin_Williams_0001.png')
image = tf.image.decode_png(file)
image.set_shape((250,250,3))
data = tf.image.per_image_standardization(image)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(file)
    print sess.run(image)
    sio.savemat('standardized_image.mat',{'image':data.eval()})
