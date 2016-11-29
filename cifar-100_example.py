###############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  example of how to use the cifar-100 input pipeline
# Date:         11.2016
#
# TODO: How can we prevent the Enqueue operation was cancelled error?

import tensorflow as tf
import time

sess = tf.Session()

input_image_batch = tf.placeholder(tf.float32, shape=[256, 32, 32, 3], name="input_image_batch")
input_label_batch = tf.placeholder(tf.float32, shape=[None, 100], name="input_label_batch")

with tf.device('/cpu:0'):
  from datasets.cifar100 import cifar100_data
  d = cifar100_data(batch_size=256)
  image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()


for i in range(10):
  print("batch ", i)
  image_batch, target_batch = d.sess.run([image_batch_tensor, target_batch_tensor])

  print(image_batch.shape)
  print(target_batch.shape)

  res = sess.run(input_image_batch, feed_dict={input_image_batch: image_batch,
                                               input_label_batch: target_batch})
  print(type(res))


print("done!")
print("Well, almost. Closing the queue and the session. This will lead to the following warning/error ...")
time.sleep(8)
d.close()

exit()
