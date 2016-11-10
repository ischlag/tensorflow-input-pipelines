###############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  example of how to use the imagenet input pipeline
# Date:         11.2016
#
# TODO: How can we prevent the Enqueue operation was cancelled error?

import tensorflow as tf
import time

sess = tf.Session()

with tf.device('/cpu:0'):
  from datasets.imagenet import imagenet_data
  d = imagenet_data(batch_size=64, sess=sess)
  image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()

for i in range(10):
  print("batch ", i)
  image_batch, target_batch = sess.run([image_batch_tensor, target_batch_tensor])
  print(image_batch.shape)
  print(target_batch.shape)

print("done!")
print("Well, almost. Closing the queue and the session. This will lead to the following warning/error ...")
time.sleep(8)
d.close()
sess.close()
exit()