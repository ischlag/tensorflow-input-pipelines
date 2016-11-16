###############################################################################
##############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  MNIST input pipeline
# Date:         11.2016
#
# Note: Only uses one queue,

""" Usage:
import tensorflow as tf

with tf.device('/cpu:0'):
  from datasets.mnist import mnist_data
  data = mnist_data(53)
  image_batch_tensor, target_batch_tensor = data.build_train_data_tensor()

sess = tf.Session()
init_op = tf.initialize_all_variables()
sess.run(init_op)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

image_batch, target_batch = sess.run([image_batch_tensor, target_batch_tensor])
print(image_batch.shape)
print(target_batch.shape)
"""

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data

class mnist_data:
  """
  Downloads the MNIST dataset and creates an input pipeline ready to be fed into a model.

  - Reshapes flat images into 28 x 28
  - converts [0 1] to [-1 1]
  - shuffles the input
  - builds batches
  """
  NUM_THREADS = 8
  NUMBER_OF_CLASSES = 10
  IMAGE_WIDTH = 28
  IMAGE_HEIGHT = 28
  NUM_OF_CHANNELS = 1

  def __init__(self, batch_size):
    """ Downloads the mnist data if necessary. """
    print("Loading MNIST data")
    self.batch_size = batch_size
    self.mnist = input_data.read_data_sets('data/MNIST', one_hot=True)

    self.TRAIN_SET_SIZE = self.mnist.train.images.shape[0]
    self.TEST_SET_SIZE = self.mnist.test.images.shape[0]
    self.VALIDATION_SET_SIZE = self.mnist.validation.images.shape[0]

  def build_train_data_tensor(self, shuffle=False, augmentation=False):
    return self.__build_generic_data_tensor(self.mnist.train.images,
                                            self.mnist.train.labels,
                                            shuffle,
                                            augmentation)

  def build_test_data_tensor(self, shuffle, augmentation=False):
    return self.__build_generic_data_tensor(self.mnist.test.images,
                                            self.mnist.test.labels,
                                            shuffle,
                                            augmentation)

  def build_validation_data_tensor(self, shuffle, augmentation=False):
    return self.__build_generic_data_tensor(self.mnist.validation.images,
                                            self.mnist.validation.labels,
                                            shuffle,
                                            augmentation)

  def __build_generic_data_tensor(self, raw_images, raw_targets, shuffle, augmentation):
    """ Creates the input pipeline and performs some preprocessing. """

    images = ops.convert_to_tensor(raw_images)
    targets = ops.convert_to_tensor(raw_targets)

    set_size = raw_images.shape[0]

    images = tf.reshape(images, [set_size, 28, 28, 1])
    image, label = tf.train.slice_input_producer([images, targets], shuffle=shuffle)

    # Data Augmentation
    if augmentation:
      # TODO
      # make sure after further preprocessing it is [0 1]
      pass

    # convert the given [0, 1] to [-1, 1]
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)

    images_batch, labels_batch = tf.train.batch([image, label], batch_size=self.batch_size, num_threads=self.NUM_THREADS)

    return images_batch, labels_batch





























