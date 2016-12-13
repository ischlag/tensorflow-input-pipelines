##############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  CIFAR-100 input pipeline
# Date:         11.2016
#
#

""" Usage:
import tensorflow as tf
sess = tf.Session()

with tf.device('/cpu:0'):
  from datasets.cifar100 import cifar100_data
  d = cifar100_data(batch_size=256, sess=sess)
  image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()

image_batch, target_batch = sess.run([image_batch_tensor, target_batch_tensor])
print(image_batch.shape)
print(target_batch.shape)
"""
import tensorflow as tf
import numpy as np
import threading

from utils import cifar100

class cifar100_data:
  """
  Downloads the CIFAR100 dataset and creates an input pipeline ready to be fed into a model.

  - Reshapes flat images into 32x32
  - converts [0 1] to [-1 1]
  - shuffles the input
  - builds batches
  """
  NUM_THREADS = 8
  NUMBER_OF_CLASSES = 100

  TRAIN_SET_SIZE = 50000
  TEST_SET_SIZE =  10000
  IMAGE_WIDTH = 32
  IMAGE_HEIGHT = 32
  NUM_OF_CHANNELS = 3

  def __init__(self, batch_size, sess,
               feed_size=200,
               feed_queue_capacity=800,
               batch_queue_capacity=1000,
               min_after_dequeue=1000):
    """ Downloads the cifar100 data if necessary. """
    print("Loading CIFAR-100 data")
    self.batch_size = batch_size
    self.feed_size = feed_size
    self.feed_queue_capacity = feed_queue_capacity
    self.batch_queue_capacity = batch_queue_capacity + 3 * batch_size
    self.min_after_dequeue = min_after_dequeue
    self.sess = sess
    cifar100.maybe_download_and_extract()

  def build_train_data_tensor(self, shuffle=False, augmentation=False):

    images, _, targets = cifar100.load_training_data()
    return self.__build_generic_data_tensor(images,
                                            targets,
                                            shuffle,
                                            augmentation)

  def build_test_data_tensor(self, shuffle=False, augmentation=False):

    images, _, targets = cifar100.load_test_data()
    return self.__build_generic_data_tensor(images,
                                            targets,
                                            shuffle,
                                            augmentation)

  def __build_generic_data_tensor(self, raw_images, raw_targets, shuffle, augmentation):
    """ Creates the input pipeline and performs some preprocessing. """

    # load the data from numpy into our queue in blocks of feed_size samples
    set_size, width, height, channels = raw_images.shape

    image_input = tf.placeholder(tf.float32, shape=[self.feed_size, width, height, channels])
    target_input = tf.placeholder(tf.float32, shape=[self.feed_size, self.NUMBER_OF_CLASSES])

    self.queue = tf.FIFOQueue(self.feed_queue_capacity, [tf.float32, tf.float32],
                         shapes=[[width, height, channels], [self.NUMBER_OF_CLASSES]])
    enqueue_op = self.queue.enqueue_many([image_input, target_input])
    image, target = self.queue.dequeue()

    # Data Augmentation
    if augmentation:
      image = tf.image.resize_image_with_crop_or_pad(image, self.IMAGE_HEIGHT+4, self.IMAGE_WIDTH+4)
      image = tf.random_crop(image, [self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.NUM_OF_CHANNELS])
      image = tf.image.random_flip_left_right(image)

    image = tf.image.per_image_standardization(image)

    if shuffle:
      images_batch, target_batch = tf.train.shuffle_batch([image, target],
                                                          batch_size=self.batch_size,
                                                          capacity=self.batch_queue_capacity,
                                                          min_after_dequeue=self.min_after_dequeue)
    else:
      images_batch, target_batch = tf.train.batch([image, target],
                                                  batch_size=self.batch_size,
                                                  capacity=self.batch_queue_capacity)

    def enqueue(sess):
      under = 0
      max = len(raw_images)
      while not self.coord.should_stop():
        upper = under + self.feed_size
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

    enqueue_thread = threading.Thread(target=enqueue, args=[self.sess])

    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

    enqueue_thread.isDaemon()
    enqueue_thread.start()

    return images_batch, target_batch

  def __del__(self):
    self.close()


  def close(self):
    self.queue.close(cancel_pending_enqueues=True)
    self.coord.request_stop()
    self.coord.join(self.threads)
    self.sess.close()