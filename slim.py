###############################################################################
#
# IDSIA, Research Project
#
# Author:       Imanol
# Description:  CNN example with clear structure.
# Date:         01.11.2016
#
#

import tensorflow as tf

from libs.components import conv2d, batch_norm, flatten, dense
from datasets.cifar10 import cifar10_data
from datasets.svhn import svhn_data

## ----------------------------------------------------------------------------
## CONFIGURATION
BATCH_SIZE = 128
LOGS_PATH = "/home/schlag/MyStuff/GrowNeurons/logs/InitialTests/4/"
EPOCHS = 300  # max number of epochs if the network never converges
learning_rate = 0.01

## ----------------------------------------------------------------------------
## DATA INPUT
sess = tf.Session()

data = cifar10_data(batch_size=BATCH_SIZE)
#data = svhn_data(batch_size=BATCH_SIZE, sess=sess)

with tf.device('/cpu:0'):
  train_image_batch, train_label_batch = data.build_train_data_tensor(shuffle=True)
  test_image_batch, test_label_batch = data.build_test_data_tensor(shuffle=False)

NUMBER_OF_CLASSES = data.NUMBER_OF_CLASSES
IMG_SIZE = data.IMAGE_SIZE
NUM_CHANNELS = data.NUM_OF_CHANNELS
TRAIN_SET_SIZE = data.TRAIN_SET_SIZE
TEST_SET_SIZE = data.TEST_SET_SIZE
TRAIN_BATCHES_PER_EPOCH = int(TRAIN_SET_SIZE / BATCH_SIZE)  # only used for training

## ----------------------------------------------------------------------------
## MODEL STRUCTURE
is_training = tf.placeholder(tf.bool, name='is_training')


def conv_block(data, n_filter, scope, stride=1):
  """An ConvBlock is a repetitive composition used in the model."""
  with tf.variable_scope(scope):
    conv = conv2d(data, n_filter, "conv",
                  k_h=3, k_w=3,
                  stride_h=stride, stride_w=stride,
                  initializer=tf.random_normal_initializer(stddev=0.01),
                  bias=True,
                  padding='SAME')
    norm = batch_norm(conv, n_filter, is_training, scope="bn")
    relu = tf.nn.relu(norm)

    conv2 = conv2d(relu, n_filter, "conv2",
                   k_h=3, k_w=3,
                   stride_h=stride, stride_w=stride,
                   initializer=tf.random_normal_initializer(stddev=0.01),
                   bias=True,
                   padding='SAME')
    norm2 = batch_norm(conv2, n_filter, is_training, scope="bn2")
    relu2 = tf.nn.relu(norm2)

    conv3 = conv2d(relu2, n_filter, "conv3",
                   k_h=3, k_w=3,
                   stride_h=stride, stride_w=stride,
                   initializer=tf.random_normal_initializer(stddev=0.01),
                   bias=True,
                   padding='SAME')
    norm3 = batch_norm(conv3, n_filter, is_training, scope="bn3")
    relu3 = tf.nn.relu(norm3)

    return relu3


def model(x):
  """Defines the CNN architecture and returns its output tensor."""
  block1 = conv_block(x, 64, "block1")
  pool1 = tf.nn.max_pool(block1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

  block2 = conv_block(pool1, 128, "block2")
  pool2 = tf.nn.max_pool(block2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

  block3 = conv_block(pool2, 256, "block3")
  pool3 = tf.nn.max_pool(block3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

  flat = flatten(pool3)

  dense1 = dense(flat, 4096, is_training, tf.nn.relu, scope="dense1", dropout=True,
                 initializer=tf.random_normal_initializer(stddev=0.01))
  dense2 = dense(dense1, 4096, is_training, tf.nn.relu, scope="dense2", dropout=True,
                 initializer=tf.random_normal_initializer(stddev=0.01))
  dense3 = dense(dense2, 4096, is_training, tf.nn.relu, scope="dense3", dropout=True,
                 initializer=tf.random_normal_initializer(stddev=0.01))
  dense6 = dense(dense3, NUMBER_OF_CLASSES, is_training, tf.nn.relu, scope="dense6", dropout=False,
                 initializer=tf.random_normal_initializer(stddev=0.01))
  return dense6


## ----------------------------------------------------------------------------
## LOSS AND ACCURACY
with tf.variable_scope("model"):
  batch_size = tf.placeholder(tf.float32, name="batch_size")

  input_image_batch = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_SIZE, IMG_SIZE, NUM_CHANNELS],
                                     name="input_image_batch")
  input_label_batch = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_CLASSES], name="input_label_batch")

  logits = model(input_image_batch)

with tf.variable_scope("loss"):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits,
    tf.cast(input_label_batch, tf.float32),
    name="cross-entropy")
  loss = tf.reduce_mean(cross_entropy, name='loss')

with tf.variable_scope("accuracy"):
  top_1_correct = tf.nn.in_top_k(logits, tf.argmax(input_label_batch, 1), 1)
  top_n_correct = tf.nn.in_top_k(logits, tf.argmax(input_label_batch, 1), 3)

predictions = tf.argmax(logits, 1)
label_batch_id = tf.argmax(input_label_batch, 1)

## ----------------------------------------------------------------------------
## OPTIMIZER
global_step = tf.get_variable('global_step', [],
                              initializer=tf.constant_initializer(0),
                              trainable=False)
lr = tf.placeholder(tf.float32, name="learning_rate")
train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(loss, global_step=global_step)

## ----------------------------------------------------------------------------
## SUMMARIES
summary_op = tf.merge_all_summaries()

## ----------------------------------------------------------------------------
## INITIALIZATION
init_op = tf.initialize_all_variables()
writer = tf.train.SummaryWriter(LOGS_PATH, graph=tf.get_default_graph())
saver = tf.train.Saver()

# initialize queue threads
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# initialize variables
sess.run(init_op)

## ----------------------------------------------------------------------------
## HELPER FUNCTIONS
def next_feed_dic(image_batch, label_batch, train=True):
  """Fetches a mini-batch of images and labels and builds a feed-dictonary"""
  with tf.device('/cpu:0'):
    curr_image_batch, curr_label_batch = sess.run([image_batch, label_batch])

    feed_dict = {
      input_image_batch: curr_image_batch,
      input_label_batch: curr_label_batch,
      batch_size: curr_image_batch.shape[0],
      is_training.name: train,
      lr.name: learning_rate
    }
    return feed_dict

## ----------------------------------------------------------------------------
## PERFORM TRAINING

# train cycles
for j in range(EPOCHS):
  print("epoch ", j)
  print("epoch.batches curr_loss (avg_loss)")
  for i in range(TRAIN_BATCHES_PER_EPOCH):
    feed_dict = next_feed_dic(train_image_batch, train_label_batch, train=True)
    _, curr_loss, summary, step = sess.run([train_op, loss, summary_op, global_step], feed_dict=feed_dict)

    if i % 30 == 0:
      print("{:3d}.{:03d} {:.5f}".format(j, i, curr_loss))

print("done")