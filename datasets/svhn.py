###############################################################################
#
# IDSIA, Research Project
#
# Author:       Imanol
# Description:  SVHN input pipeline
# Date:         03.11.2016
#
#

""" Usage:
import tensorflow as tf
sess = tf.Session()

from datasets.svhn import svhn_data
d = svhn_data(batch_size=256, sess=sess)
tensor_images, tensor_targets = d.build_train_data_tensor()

init_op = tf.initialize_all_variables()
sess.run(init_op)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

images_batch, labels_batch = sess.run([tensor_images, tensor_targets])
"""
import tensorflow as tf
import numpy as np
import threading

from utils import svhn

class svhn_data:
  """
  Downloads the SVHN dataset and creates an input pipeline ready to be fed into a model.

  - Reshapes flat images into 32x32
  - converts [0 1] to [-1 1]
  - shuffles the input
  - builds batches
  """
  NUM_THREADS = 8
  NUMBER_OF_CLASSES = 10

  TRAIN_SET_SIZE = 73257
  TEST_SET_SIZE =  26032
  IMAGE_WIDTH = 32
  IMAGE_HEIGHT = 32
  NUM_OF_CHANNELS = 3

  def __init__(self, batch_size, sess, feed_size=200, feed_queue_capacity=800, batch_queue_capacity=5, min_after_dequeue=4):
    """ Downloads the cifar100 data if necessary. """
    self.batch_size = batch_size
    self.feed_size = feed_size
    self.feed_queue_capacity = feed_queue_capacity
    self.batch_queue_capacity = batch_queue_capacity
    self.min_after_dequeue = min_after_dequeue
    self.sess = sess
    svhn.download_data()

  def build_train_data_tensor(self, shuffle=False, augmentation=False):
    images, _, targets = svhn.load_training_data()
    return self.__build_generic_data_tensor(images,
                                            targets,
                                            shuffle,
                                            augmentation)

  def build_test_data_tensor(self, shuffle=False, augmentation=False):
    images, _, targets = svhn.load_test_data()
    return self.__build_generic_data_tensor(images,
                                            targets,
                                            shuffle,
                                            augmentation)

  def __build_generic_data_tensor(self, raw_images, raw_targets, shuffle, augmentation):
    """
    Creates the input pipeline and performs some preprocessing.
    The full dataset needs to fit into memory for this version.
    """

    # load the data from numpy into our queue in blocks of feed_size samples
    set_size, width, height, channels = raw_images.shape

    image_input = tf.placeholder(tf.float32, shape=[self.feed_size, width, height, channels])
    target_input = tf.placeholder(tf.float32, shape=[self.feed_size, self.NUMBER_OF_CLASSES])

    queue = tf.FIFOQueue(self.feed_queue_capacity, [tf.float32, tf.float32],
                         shapes=[[width, height, channels], [self.NUMBER_OF_CLASSES]])
    enqueue_op = queue.enqueue_many([image_input, target_input])
    image, target = queue.dequeue()

    # Data Augmentation
    if augmentation:
      # TODO
      # make sure after further preprocessing it is [0 1]
      pass

    # convert the given [0, 1] to [-1, 1]
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)

    images_batch, target_batch = tf.train.shuffle_batch([image, target],
                                                        batch_size=self.batch_size,
                                                        capacity=self.batch_queue_capacity,
                                                        min_after_dequeue=self.min_after_dequeue)
    #images_batch, target_batch = tf.train.batch([image, target],
    #                                            batch_size=self.batch_size,
    #                                            capacity=self.batch_queue_capacity)

    #run_options = tf.RunOptions(timeout_in_ms=4000)

    def enqueue(sess):
      under = 0
      max = len(raw_images)
      while True:
        #print("starting to write into queue")
        upper = under + self.feed_size
        #print("try to enqueue ", under, " to ", upper)
        if upper <= max:
          curr_data = raw_images[under:upper]
          curr_target = raw_targets[under:upper]
          under = upper
        else:
          rest = upper - max
          curr_data = np.concatenate((raw_images[under:max], raw_images[0:rest]))
          curr_target = np.concatenate((raw_targets[under:max], raw_targets[0:rest]))
          under = rest

        sess.run(enqueue_op, feed_dict={image_input: curr_data,
                                        target_input: curr_target})
        #print("added!")

    enqueue_thread = threading.Thread(target=enqueue, args=[self.sess])

    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

    enqueue_thread.isDaemon()
    enqueue_thread.start()

    return images_batch, target_batch

  def __exit__(self, exc_type, exc_value, traceback):
    self.coord.request_stop()
    self.coord.join(self.threads)
    self.sess.close()


