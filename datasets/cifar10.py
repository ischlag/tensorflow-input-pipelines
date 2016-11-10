###############################################################################
#
# IDSIA, Research Project
#
# Author:       Imanol
# Description:  CIFAR10 input pipeline
# Date:         01.11.2016
#
#

""" Usage:
from datasets.cifar10 import cifar10_data
data = cifar10_data(53)
tensor_images, tensor_labels = data.build_train_data_tensor()

import tensorflow as tf
sess = tf.Session()
init_op = tf.initialize_all_variables()
sess.run(init_op)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

images_batch, labels_batch = sess.run([tensor_images, tensor_labels])
print(images_batch.shape)
print(labels_batch.shape)
"""

import tensorflow as tf

from utils import cifar10
from tensorflow.python.framework import ops

class cifar10_data:
  """
  Downloads the CIFAR10 dataset and creates an input pipeline ready to be fed into a model.

  - Reshapes flat images into 32x32
  - converts [0 1] to [-1 1]
  - shuffles the input
  - builds batches
  """
  NUM_THREADS = 8
  NUMBER_OF_CLASSES = 10

  TRAIN_SET_SIZE = 50000
  TEST_SET_SIZE =  10000
  IMAGE_WIDTH = 32
  IMAGE_HEIGHT = 32
  NUM_OF_CHANNELS = 3

  def __init__(self, batch_size):
    """ Downloads the cifar10 data if necessary. """
    self.batch_size = batch_size
    cifar10.maybe_download_and_extract()

  def build_train_data_tensor(self, shuffle=False, augmentation=False):
    images, _, targets = cifar10.load_training_data()
    return self.__build_generic_data_tensor(images,
                                            targets,
                                            shuffle,
                                            augmentation)

  def build_test_data_tensor(self, shuffle=False, augmentation=False):
    images, _, targets = cifar10.load_test_data()
    return self.__build_generic_data_tensor(images,
                                            targets,
                                            shuffle,
                                            augmentation)

  def __build_generic_data_tensor(self, raw_images, raw_targets, shuffle, augmentation):
    """ Creates the input pipeline and performs some preprocessing. """

    images = ops.convert_to_tensor(raw_images)
    targets = ops.convert_to_tensor(raw_targets)

    set_size, width, height, channels = raw_images.shape

    images = tf.reshape(images, [set_size, width, height, channels])
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


