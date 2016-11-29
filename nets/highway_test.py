# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Build a Highway network on ImageNet data set.

Summary of available functions:
 inference: Compute inference on the model inputs to make a prediction
 loss: Compute the loss of the prediction with respect to the labels
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from libs import custom_ops

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.0

def inference(images, num_classes, is_training):
  # Parameters for BatchNorm.
  sizes = (3, 4, 6, 3)
  filters = (45, 90, 180, 360)

  with tf.variable_scope('stage1'):
    conv1 = slim.conv2d(images,
                        num_outputs=64,
                        kernel_size=(7, 7),
                        stride=2,
                        padding='SAME',
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                          factor=2.0, mode='FAN_OUT', uniform=False),
                        weights_regularizer=slim.l2_regularizer(0.0001),
                        activation_fn=None)
    norm1 = slim.batch_norm(conv1, decay=0.9, epsilon=0.00005, is_training=is_training)
    relu1 = tf.nn.relu(norm1)
    pool1 = slim.max_pool2d(relu1, kernel_size=[3, 3], stride=2, padding='SAME')


  m, n = sizes[0] - 1, filters[0]
  with tf.variable_scope('stage2'):
    hw2a = custom_ops.hwblock(pool1, num_filters=n, weighted_skip=True, is_training=is_training)
    hw2x = slim.repeat(hw2a, m, custom_ops.hwblock, num_filters=n)

  '''
    m, n = sizes[1] - 1, filters[1]
    with tf.variable_scope('stage3'):
      hw3a = slim.ops.hwblock(hw2x, num_filters=n, mid_stride=2)
      hw3x = slim.ops.repeat_op(m, hw3a, slim.ops.hwblock, num_filters=n)

    m, n = sizes[2] - 1, filters[2]
    with tf.variable_scope('stage4'):
      hw4a = slim.ops.hwblock(hw3x, num_filters=n, mid_stride=2)
      hw4x = slim.ops.repeat_op(m, hw4a, slim.ops.hwblock, num_filters=n)

    m, n = sizes[3] - 1, filters[3]
    with tf.variable_scope('stage5'):
      hw5a = slim.ops.hwblock(hw4x, num_filters=n, mid_stride=2)
      hw5x = slim.ops.repeat_op(m, hw5a, slim.ops.hwblock, num_filters=n)

    pool = slim.ops.avg_pool(hw5x, kernel_size=(7, 7), stride=1, scope='pool')
  '''

  net = slim.layers.flatten(hw2x, scope='flatten')
  bias_lim = 1.0/np.sqrt(1440)
  logits = slim.layers.fully_connected(net, num_classes, activation_fn=None, scope='logits',
                                       biases_initializer=tf.random_uniform_initializer(-bias_lim, bias_lim))

  return logits
