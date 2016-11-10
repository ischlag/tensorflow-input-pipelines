###############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  imagenet input pipeline
# Date:         11.2016
#
#
# TODO: 23 images are not jpeg and should be used with the according decoder.

""" Usage:
import tensorflow as tf
sess = tf.Session()

with tf.device('/cpu:0'):
  from datasets.imagenet import imagenet_data
  d = imagenet_data(batch_size=64, sess=sess)
  image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()

image_batch, target_batch = sess.run([image_batch_tensor, target_batch_tensor])
print(image_batch.shape)
print(target_batch.shape)
"""

import tensorflow as tf
import numpy as np
import threading

from utils import imagenet

class imagenet_data:
  """
  Downloads the imagenet dataset and creates an input pipeline ready to be fed into a model.

  memory calculation:
    1 image is 299*299*3*4 bytes = ~1MB
    1024MB RAM = ~1000 images

  empirical memory usage with default config:
    TensorFlow +500MB
    imagenet_utils (loading all paths and labels) +400MB
    build input pipeline and fill queues +2.2GB

  - decodes jpg images
  - scales images into a uniform size
  - shuffles the input if specified
  - builds batches
  """
  NUM_THREADS = 8
  NUMBER_OF_CLASSES = 1000
  TRAIN_SET_SIZE = len(imagenet.data.train_filenames) # 1281167 # ~250MB for string with paths
  TEST_SET_SIZE = len(imagenet.data.val_filenames) # 50000
  IMAGE_HEIGHT = 299
  IMAGE_WIDTH = 299
  NUM_OF_CHANNELS = 3

  def __init__(self, batch_size, sess,
               filename_feed_size=200,
               filename_queue_capacity=800,
               batch_queue_capacity=1000,
               min_after_dequeue=1000,
               image_height=IMAGE_HEIGHT,
               image_width=IMAGE_WIDTH):
    """ Downloads the data if necessary. """
    self.batch_size = batch_size
    self.filename_feed_size = filename_feed_size
    self.filename_queue_capacity = filename_queue_capacity
    self.batch_queue_capacity = batch_queue_capacity + 3 * batch_size
    self.min_after_dequeue = min_after_dequeue
    self.sess = sess
    self.IMAGE_HEIGHT = image_height
    self.IMAGE_WIDTH = image_width
    imagenet.check_if_downloaded()

  def build_train_data_tensor(self, shuffle=False, augmentation=False):
    img_path, cls = imagenet.load_training_data()
    return self.__build_generic_data_tensor(img_path, cls, shuffle, augmentation)

  def build_test_data_tensor(self, shuffle=False, augmentation=False):
    img_path, cls = imagenet.load_test_data()
    return self.__build_generic_data_tensor(img_path, cls, shuffle, augmentation)

  def __build_generic_data_tensor(self, all_img_paths, all_targets, shuffle, augmentation):
    """
    Creates the input pipeline and performs some preprocessing.
    The full dataset needs to fit into memory for this version.
    """

    ## filename queue
    imagepath_input = tf.placeholder(tf.string, shape=[self.filename_feed_size])
    target_input = tf.placeholder(tf.float32, shape=[self.filename_feed_size])

    self.filename_queue = tf.FIFOQueue(self.filename_queue_capacity, [tf.string, tf.float32],
                                  shapes=[[], []])
    enqueue_op = self.filename_queue.enqueue_many([imagepath_input, target_input])
    single_path, single_target = self.filename_queue.dequeue()

    # one hot encode the target
    single_target = tf.cast(tf.sub(single_target, tf.constant(1.0)), tf.int32)
    single_target = tf.one_hot(single_target, depth=self.NUMBER_OF_CLASSES)

    # load the jpg image according to path
    file_content = tf.read_file(single_path)
    single_image = tf.image.decode_jpeg(file_content, channels=self.NUM_OF_CHANNELS)

    # convert to [0, 1]
    single_image = tf.image.convert_image_dtype(single_image,
                                                dtype=tf.float32,
                                                saturate=True)

    single_image = tf.image.resize_images(single_image, [self.IMAGE_HEIGHT, self.IMAGE_WIDTH])

    # Data Augmentation
    if augmentation:
      # TODO
      # make sure after further preprocessing it is still [0 1]
      pass

    # convert the given [0, 1] to [-1, 1]
    single_image = tf.sub(single_image, 0.5)
    single_image = tf.mul(single_image, 2.0)

    if shuffle:
      images_batch, target_batch = tf.train.shuffle_batch([single_image, single_target],
                                                          batch_size=self.batch_size,
                                                          capacity=self.batch_queue_capacity,
                                                          min_after_dequeue=self.min_after_dequeue,
                                                          num_threads=self.NUM_THREADS)
    else:
      images_batch, target_batch = tf.train.batch([single_image, single_target],
                                                          batch_size=self.batch_size,
                                                          capacity=self.batch_queue_capacity,
                                                          num_threads=1)

    def enqueue(sess):
      under = 0
      max = len(all_img_paths)
      while not self.coord.should_stop():
        upper = under + self.filename_feed_size
        if upper <= max:
          curr_data = all_img_paths[under:upper]
          curr_target = all_targets[under:upper]
          under = upper
        else:
          rest = upper - max
          curr_data = np.concatenate((all_img_paths[under:max], all_img_paths[0:rest]))
          curr_target = np.concatenate((all_targets[under:max], all_targets[0:rest]))
          under = rest

        sess.run(enqueue_op, feed_dict={imagepath_input: curr_data,
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
    self.filename_queue.close(cancel_pending_enqueues=True)
    self.coord.request_stop()
    self.coord.join(self.threads)







































