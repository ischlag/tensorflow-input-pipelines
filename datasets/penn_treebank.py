##############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  penn treebank input pipeline
# Date:         11.2016
#
# Note: Code mostly from the TensorFlow ptb example but with automatic download.
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py

""" Usage:
import tensorflow as tf
sess = tf.Session()

with tf.device('/cpu:0'):
  from datasets.penn_treebank import penn_treebank_data
  d = penn_treebank_data(batch_size=2, num_steps=5)
  image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

image_batch, target_batch = sess.run([image_batch_tensor, target_batch_tensor])
print(image_batch)
print(target_batch)
"""

import tensorflow as tf

from utils import penn_treebank

class penn_treebank_data:
  """
  Downloads the stanford cars 196 dataset and creates an input pipeline ready to be fed into a model.

  - decodes jpg images
  - scales images into a uniform size
  - shuffles the input
  - builds batches
  """

  NUM_THREADS = 8

  def __init__(self, batch_size, num_steps):   # 100MB RAM ~=1150 images
    """ Downloads the data if necessary. """
    self.batch_size = batch_size
    self.num_steps = num_steps
    penn_treebank.download_data()

  def build_train_data_tensor(self):
    data, _ = penn_treebank.load_training_data()
    return self.__build_generic_data_tensor(data)

  def build_test_data_tensor(self):
    data, _ = penn_treebank.load_training_data()
    return self.__build_generic_data_tensor(data)

  def build_validation_data_tensor(self):
    data, _ = penn_treebank.load_validation_data()
    return self.__build_generic_data_tensor(data)

  def __build_generic_data_tensor(self, raw_data):
    """Iterate on the raw PTB data.
    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.
    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
      name: the name of this operation (optional).
    Returns:
      A pair of Tensors, each shaped [batch_size, num_steps]. The second element
      of the tuple is the same data time-shifted to the right by one.
    Raises:
      tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // self.batch_size
    data = tf.reshape(raw_data[0: self.batch_size * batch_len],
                      [self.batch_size, batch_len])

    epoch_size = (batch_len - 1) // self.num_steps
    assertion = tf.assert_positive(
      epoch_size,
      message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.slice(data, [0, i * self.num_steps], [self.batch_size, self.num_steps])
    y = tf.slice(data, [0, i * self.num_steps + 1], [self.batch_size, self.num_steps])
    return x, y

