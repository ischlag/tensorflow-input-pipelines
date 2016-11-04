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
import numpy as np
import shutil
import os
import time

from collections import deque
from libs.components import conv2d, batch_norm, flatten, dense
from datasets.mnist import mnist_data
from datasets.cifar10 import cifar10_data
from datasets.cifar100 import cifar100_data
from datasets.svhn import svhn_data

## ----------------------------------------------------------------------------
## CONFIGURATION
BATCH_SIZE = 128
LOGS_PATH = "/home/schlag/MyStuff/GrowNeurons/logs/InitialTests/4/"

LOAD_PARAMS = False
PARAM_PATH = "/home/schlag/MyStuff/GrowNeurons/logs/InitialTests/3/"
ONLY_EVAL = False # If True, no training is performed
# ---
EPOCHS = 300 # max number of epochs if the network never converges
learning_rate = 0.01
DECREASE_LEARNING_RATE_AFTER_N_BAD_EPOCHS = 5
DECREASE_LEARNING_RATE_N_TIMES = 3
SAVE_AFTER_MIN_N_EPOCHS = -1
LEARNING_RATE_DECAY_FACTOR = 2.0

## ----------------------------------------------------------------------------
## DATA INPUT
sess = tf.Session()

#data = mnist_data(batch_size=BATCH_SIZE)
#data = cifar10_data(batch_size=BATCH_SIZE)
#data = cifar100_data(batch_size=BATCH_SIZE)
data = svhn_data(batch_size=BATCH_SIZE, sess=sess)

with tf.device('/cpu:0'):
  train_image_batch, train_label_batch = data.build_train_data_tensor(shuffle=True)
  test_image_batch, test_label_batch = data.build_test_data_tensor(shuffle=False)

NUMBER_OF_CLASSES = data.NUMBER_OF_CLASSES
IMG_SIZE = data.IMAGE_SIZE
NUM_CHANNELS = data.NUM_OF_CHANNELS
TRAIN_SET_SIZE = data.TRAIN_SET_SIZE
TEST_SET_SIZE = data.TEST_SET_SIZE
#VALID_SET_SIZE = data.validation_set_size
TRAIN_BATCHES_PER_EPOCH = int(TRAIN_SET_SIZE / BATCH_SIZE) # only used for training

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

  input_image_batch = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], name="input_image_batch")
  input_label_batch = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_CLASSES], name="input_label_batch")

  logits = model(input_image_batch)

with tf.variable_scope("loss"):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                          logits,
                          tf.cast(input_label_batch, tf.float32),
                          name="cross-entropy")
  loss = tf.reduce_mean(cross_entropy, name='loss')
  
with tf.variable_scope("accuracy"):
  top_1_correct = tf.nn.in_top_k(logits, tf.argmax(input_label_batch,1), 1)
  top_n_correct = tf.nn.in_top_k(logits, tf.argmax(input_label_batch,1), 3)
  top_1_batch_accuracy = tf.reduce_sum(tf.cast(top_1_correct, tf.float32)) * 100.0 / batch_size
  top_n_batch_accuracy = tf.reduce_sum(tf.cast(top_n_correct, tf.float32)) * 100.0 / batch_size

predictions = tf.argmax(logits,1)
label_batch_id = tf.argmax(input_label_batch,1)

## ----------------------------------------------------------------------------
## OPTIMIZER
global_step = tf.get_variable('global_step', [],
                  initializer = tf.constant_initializer(0),
                  trainable = False)
lr = tf.placeholder(tf.float32, name="learning_rate")
train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(loss, global_step=global_step)

## ----------------------------------------------------------------------------
## SUMMARIES
tf.scalar_summary("train/loss", loss)
tf.scalar_summary("train/top_1_batch_acc", top_1_batch_accuracy)
tf.scalar_summary("train/top_n_batch_acc", top_n_batch_accuracy)
tf.scalar_summary("train/learning_rate", lr)
train_summary_op = tf.merge_all_summaries()

a = tf.scalar_summary("test/loss", loss)
b = tf.scalar_summary("test/top_1_batch_acc", top_1_batch_accuracy)
c = tf.scalar_summary("test/top_n_batch_acc", top_n_batch_accuracy)
test_summary_op = tf.merge_summary([a, b, c])

a = tf.scalar_summary("validation/loss", loss)
b = tf.scalar_summary("validation/top_1_batch_acc", top_1_batch_accuracy)
c = tf.scalar_summary("validation/top_n_batch_acc", top_n_batch_accuracy)
valid_summary_op = tf.merge_summary([a, b, c])

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

# load parameters from a save file
if LOAD_PARAMS:
  saver.restore(sess, PARAM_PATH + "best-model.ckpt")
  print("params loaded!")

## ----------------------------------------------------------------------------
## HELPER FUNCTIONS
def print_confusion(confusion_matrix):
  """Print to the console the confusion matrix"""
  print("confusion: (count, label)")
  for i in range(len(confusion_matrix)):
    idx = []
    for j in range(len(confusion_matrix[i])):
      if confusion_matrix[i][j] != 0 and j != i:
        idx.append((confusion_matrix[i][j], j))
    idx = sorted(idx, key=lambda tup: tup[0], reverse=True)
    print("label", i, " most mistaken for: ", idx)

def print_label_accuracy(top_1_res, top_n_res):
  """Prints the pre class accuracy nicely to the console"""
  print("Accuracy:               top-1 | top-n")
  for e in top_1_res:
    top_1_percent = ((100.0 / top_1_res[e][1]) * top_1_res[e][0]) if top_1_res[e][1] != 0.0 else 0.0
    top_n_percent = ((100.0 / top_n_res[e][1]) * top_n_res[e][0]) if top_n_res[e][1] != 0.0 else 0.0

    print("Label {:2d}: {:4d} {:4d} ({: >6.2f}%) | {:4d} {:4d} ({: >6.2f}%) ".format(e, top_1_res[e][0], top_1_res[e][1],
                                                                               top_1_percent,
                                                                               top_n_res[e][0], top_n_res[e][1],
                                                                               top_n_percent))

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

def eval_accuracy(top_1_correct, top_n_correct, label_batch_id, predictions, batch_size, set_size, image_batch,
                  label_batch, summary_op, global_step):
  """Performs one epoch (no training) and calculates different performance statistics"""

  def log_results(curr_top_1_correct, curr_top_n_correct, curr_label_id, curr_prediction):
    """Helper function used in eval_similarity"""
    # increase counter for every sample
    for i in range(len(curr_label_id)):
      eval_accuracy.confusion[curr_label_id[i]][curr_prediction[i]] += 1

    # increase positive and negative count for top 1 prediction
    for i in range(len(curr_top_1_correct)):
      if curr_top_1_correct[i]:
        eval_accuracy.top_1_results[curr_label_id[i]][0] += 1
      eval_accuracy.top_1_results[curr_label_id[i]][1] += 1

    # increase positive and negative count for top n prediction
    for i in range(len(curr_top_n_correct)):
      if curr_top_n_correct[i]:
        eval_accuracy.top_n_results[curr_label_id[i]][0] += 1
      eval_accuracy.top_n_results[curr_label_id[i]][1] += 1

    # increase the
    eval_accuracy.top_1_true_count += np.sum(curr_top_1_correct)
    eval_accuracy.top_n_true_count += np.sum(curr_top_n_correct)

  # actual number of positives and negatives for top 1 prediction
  eval_accuracy.top_1_results = {}
  for i in range(NUMBER_OF_CLASSES):
    eval_accuracy.top_1_results[i] = [0, 0]  # [true_count, false_count]

  # actual number of positives and negatives for top n prediction
  eval_accuracy.top_n_results = {}
  for i in range(NUMBER_OF_CLASSES):
    eval_accuracy.top_n_results[i] = [0, 0]  # [true_count, false_count]

  # confusion matrix
  eval_accuracy.confusion = []
  for i in range(NUMBER_OF_CLASSES):
    eval_accuracy.confusion.append(NUMBER_OF_CLASSES * [0])

  # accuracy
  eval_accuracy.top_1_true_count = 0.0
  eval_accuracy.top_n_true_count = 0.0

  # iterate over full batches
  batches = int(set_size / batch_size)
  rest = set_size - (batches * batch_size)
  for i in range(batches):
    feed_dict = next_feed_dic(image_batch, label_batch, train=False)
    curr_top_1_correct, curr_top_n_correct, curr_label_id, curr_prediction, summary, step = sess.run(
      [top_1_correct, top_n_correct, label_batch_id, predictions, summary_op, global_step], feed_dict)
    log_results(curr_top_1_correct, curr_top_n_correct, curr_label_id, curr_prediction)
    writer.add_summary(summary, step)

  # evaluate last batch and remove spare
  if rest > 0:
    feed_dict = next_feed_dic(image_batch, label_batch, train=False)
    curr_top_1_correct, curr_top_n_correct, curr_label_id, curr_prediction, summary, step = sess.run(
      [top_1_correct, top_n_correct, label_batch_id, predictions, summary_op, global_step], feed_dict)

    # remove samples from next epoch
    curr_top_1_correct = curr_top_1_correct[:rest]
    curr_top_n_correct = curr_top_n_correct[:rest]
    curr_label_id = curr_label_id[:rest]
    curr_prediction = curr_prediction[:rest]

    log_results(curr_top_1_correct, curr_top_n_correct, curr_label_id, curr_prediction)
    writer.add_summary(summary, step)

  return eval_accuracy.top_1_true_count / set_size, eval_accuracy.top_n_true_count / set_size, eval_accuracy.top_1_results, eval_accuracy.top_n_results, eval_accuracy.confusion

def test_accuracy(display_confusion=False):
  """Calculates the accuracy for the test set and prints it to the console."""
  top_1_acc, top_n_acc, top_1_res, top_n_res, confusion = eval_accuracy(top_1_correct,
                                                                        top_n_correct,
                                                                        label_batch_id,
                                                                        predictions,
                                                                        BATCH_SIZE,
                                                                        TEST_SET_SIZE,
                                                                        test_image_batch,
                                                                        test_label_batch,
                                                                        test_summary_op,
                                                                        global_step)
  print("test set accuracy top-1: {:6.3f}%({:6.3f}%) top-n: {:6.3f}%({:6.3f}%)".format(
    top_1_acc*100,(1.0-top_1_acc)*100.0, top_n_acc*100.0, (1.0-top_n_acc)*100.0))
  if display_confusion:
    print_label_accuracy(top_1_res, top_n_res)
    print_confusion(confusion)
  return top_1_acc

'''
def validation_accuracy(display_confusion=False):
  """Calculates the accuracy for the validation set and prints it to the console."""
  top_1_acc, top_n_acc, top_1_res, top_n_res, confusion = eval_accuracy(top_1_correct,
                                                                        top_n_correct,
                                                                        label_batch_id,
                                                                        predictions,
                                                                        BATCH_SIZE,
                                                                        VALID_SET_SIZE,
                                                                        eval_validation_image_batch,
                                                                        eval_validation_label_batch,
                                                                        valid_summary_op,
                                                                        global_step)
  print("test set accuracy top-1: {:6.3f}%({:6.3f}%) top-n: {:6.3f}%({:6.3f}%)".format(
    top_1_acc*100,(1.0-top_1_acc)*100.0, top_n_acc*100.0, (1.0-top_n_acc)*100.0))
  if display_confusion:
    print_label_accuracy(top_1_res, top_n_res)
    print_confusion(confusion)
  return top_1_acc
'''

def train_accuracy(display_confusion=False):
  """Calculates the accuracy for the train set, prints it to the console."""
  top_1_acc, top_n_acc, top_1_res, top_n_res, confusion = eval_accuracy(top_1_correct,
                                                                        top_n_correct,
                                                                        label_batch_id,
                                                                        predictions,
                                                                        BATCH_SIZE,
                                                                        TRAIN_SET_SIZE,
                                                                        train_image_batch,
                                                                        train_label_batch,
                                                                        train_summary_op,
                                                                        global_step)
  print("train set accuracy top-1: {:6.3f}%({:6.3f}%) top-n: {:6.3f}%({:6.3f}%)".format(
        top_1_acc*100,(1.0-top_1_acc)*100.0, top_n_acc*100.0, (1.0-top_n_acc)*100.0))
  if display_confusion:
    print_label_accuracy(top_1_res, top_n_res)
    print_confusion(confusion)
  return top_1_acc

def push_into_queue(value, queue, tag, step):
  """Pushes new values into a queue of fixed length and writes the average of that queue into a summary operation."""
  queue.pop()
  queue.appendleft(value)
  avg = np.mean(queue).item()
  avg_summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=avg)])
  writer.add_summary(avg_summary, global_step=step)
  return avg

## ----------------------------------------------------------------------------
## ONLY PERFORM EVALUATION
if ONLY_EVAL:
  print("Only Evaluate:")
  train_accuracy()
  test_accuracy()
  #validation_accuracy()
  exit()

## ----------------------------------------------------------------------------
## PERFORM TRAINING
# save this script in the log folder
shutil.copy2(os.path.realpath(__file__), LOGS_PATH)

# queues for the averages
train_loss_queue = deque(TRAIN_BATCHES_PER_EPOCH * [0])

# train cycles
best_accuracy = 0
accuracy_not_increased_counter = 0
lr_reduction_counter = 0
start_session_time = time.time()
for j in range(EPOCHS):
  print("epoch ", j)
  print("epoch.batches curr_loss (avg_loss)")
  for i in range(TRAIN_BATCHES_PER_EPOCH):
    feed_dict = next_feed_dic(train_image_batch, train_label_batch, train=True)
    _, curr_loss, summary, step = sess.run([train_op, loss, train_summary_op, global_step], feed_dict=feed_dict)

    avg_trloss = push_into_queue(curr_loss, train_loss_queue, "train/avg_loss", step)
    writer.add_summary(summary, step)

    if i % 30 == 0:
      print("{:3d}.{:03d} {:.5f} ({:.5f})".format(j, i, curr_loss, avg_trloss))

  elapsed_session_time = (time.time() - start_session_time) / 60.0
  print("time: {:7.2f}min".format(elapsed_session_time))

  # save best model
  curr_acc = test_accuracy(display_confusion=False)
  if j > SAVE_AFTER_MIN_N_EPOCHS:
    if curr_acc > best_accuracy:
      best_accuracy = curr_acc
      accuracy_not_increased_counter = 0
      print("best test accuracy -> saving parameters to best-model.ckpt")
      saver.save(sess, LOGS_PATH + "best-model.ckpt")
    else:
      accuracy_not_increased_counter += 1

  # decrease learning rate if learning stagnates, and stop if necessary
  if accuracy_not_increased_counter >= DECREASE_LEARNING_RATE_AFTER_N_BAD_EPOCHS:
    accuracy_not_increased_counter = 0
    learning_rate /= LEARNING_RATE_DECAY_FACTOR
    lr_reduction_counter += 1
    if lr_reduction_counter > DECREASE_LEARNING_RATE_N_TIMES:
      print("LEARNING RATE DECREASED a {}. time. \nTRAINING STOPS.\n".format(lr_reduction_counter))
      saver.restore(sess, LOGS_PATH + "best-model.ckpt")
      print("best model loaded!")
      train_accuracy(display_confusion=True)
      test_accuracy(display_confusion=True)
      #validation_accuracy(display_confusion=True)
      exit()
    print("LEARNING RATE DECREASED to ", learning_rate)

print("done")