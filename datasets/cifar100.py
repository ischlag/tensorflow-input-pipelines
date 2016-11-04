###############################################################################
#
# IDSIA, Research Project
#
# Author:       Imanol
# Description:  CIFAR100 input pipeline
# Date:         02.11.2016
#
#

""" Usage:

"""
import tensorflow as tf

from utils import cifar100
from tensorflow.python.framework import ops

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
  IMAGE_SIZE = 32
  NUM_OF_CHANNELS = 3

  def __init__(self, batch_size):
    """ Downloads the cifar100 data if necessary. """
    self.batch_size = batch_size
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
