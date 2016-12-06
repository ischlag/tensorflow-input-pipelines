"""ResNet model implemented using slim components

Related ResNet papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.training import moving_averages

HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')


class ResNet(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.merge_all_summaries()

  def _build_model(self):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
      x = self._images
      x = self._conv('init_conv', x, 16, stride=1)

    res_func = self._residual
    filters = [16, 16, 32, 64]
    # Uncomment the following codes to use w28-10 wide residual network.
    # It is more memory efficient than very deep residual network and has
    # comparably good performance.
    # https://arxiv.org/pdf/1605.07146v1.pdf
    # filters = [16, 160, 320, 640]
    # Update hps.num_residual_units to 9

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], stride=1)
    for i in range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], stride=1)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], stride=2)
    for i in range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], stride=1)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], stride=2)
    for i in range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], stride=1)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm(x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      x = slim.layers.flatten(x)
      self.logits = self._fully_connected(x, self.hps.num_classes)
      self.predictions = tf.nn.softmax(self.logits)

    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          self.logits, self.labels)
      self.cost = tf.reduce_mean(xent, name='xent')
      self.cost += self._decay()

      tf.scalar_summary('cost', self.cost)

  def _residual(self, x, in_filter, out_filter, stride):
    """Residual unit with 2 sub layers."""
    with tf.variable_scope('residual_only_activation'):
      orig_x = x
      x = self._batch_norm(x)
      x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, out_filter, stride=stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm(x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, out_filter, stride=1)

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.info('image after unit %s', x.get_shape())
    return x

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.scalar_summary('learning rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops + tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    self.train_op = tf.group(*train_ops)

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'weights') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.histogram_summary(var.op.name, var)

    return tf.mul(self.hps.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, out_filters, stride):
    filter_size = 3
    n = filter_size * filter_size * out_filters
    return slim.layers.conv2d(x, out_filters, [filter_size, filter_size], stride=stride,
                              padding='SAME', activation_fn=None,
                              weights_initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)),
                              scope=name)

  def _batch_norm(self, x):
    if self.mode == 'train':
      return slim.layers.batch_norm(x, scale=False, decay=0.9, scope='bn', is_training=True)
    else:
      return slim.layers.batch_norm(x, scale=False, decay=0.9, scope='bn', is_training=False)


  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    return slim.layers.fully_connected(x, out_dim,
                                weights_initializer=tf.uniform_unit_scaling_initializer(factor=1.0))

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])