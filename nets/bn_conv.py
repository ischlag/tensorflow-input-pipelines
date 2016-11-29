import tensorflow as tf
import tensorflow.contrib.slim as slim

def conv_block(data, n_filter, is_training, scope, stride=1):
  """An ConvBlock is a repetitive composition used in the model."""
  with tf.variable_scope(scope):
    conv1 = slim.layers.conv2d(data, n_filter, [3, 3], stride=stride, padding='SAME', activation_fn=None)
    norm1 = slim.layers.batch_norm(conv1, scale=False, decay=0.5, epsilon=0.001, is_training=is_training)
    relu1 = tf.nn.relu(norm1)

    conv2 = slim.layers.conv2d(relu1, n_filter, [3, 3], stride=stride, padding='SAME', activation_fn=None)
    norm2 = slim.layers.batch_norm(conv2, scale=False, decay=0.5, epsilon=0.001, is_training=is_training)
    relu2 = tf.nn.relu(norm2)

    return relu2

def inference(images, num_classes, is_training):
  """Defines the architecture and returns logits."""
  block1 = conv_block(images, 64, is_training, "block1")
  pool1 = tf.nn.max_pool(block1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

  block2 = conv_block(pool1, 128, is_training, "block2")
  pool2 = tf.nn.max_pool(block2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

  block3 = conv_block(pool2, 256, is_training, "block3")
  pool3 = tf.nn.max_pool(block3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

  flat = slim.layers.flatten(pool3)

  dense1 = slim.layers.fully_connected(flat, 2048, tf.nn.relu, scope="dense1",
                                       weights_initializer=tf.random_normal_initializer(stddev=0.01))
  dense1_dropout = slim.layers.dropout(dense1, is_training=is_training)

  dense2 = slim.layers.fully_connected(dense1_dropout, 2048, tf.nn.relu, scope="dense2",
                                       weights_initializer=tf.random_normal_initializer(stddev=0.01))
  dense2_dropout = slim.layers.dropout(dense2, is_training=is_training)

  dense6 = slim.layers.fully_connected(dense2_dropout, num_classes, tf.nn.relu, scope="dense3",
                                       weights_initializer=tf.random_normal_initializer(stddev=0.01))
  return dense6