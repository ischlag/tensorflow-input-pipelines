import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
from datetime import datetime

from datasets.cifar10 import cifar10_data
from datasets.cifar100 import cifar100_data

from libs import custom_ops
from nets import bn_conv
from nets import highway_test

log_dir = "logs/cifar10/6"
batch_size = 64
num_classes = 10
epoch_in_steps = int(50000.0/batch_size)
max_step = epoch_in_steps * 15

sess = tf.Session()

## Data
with tf.device('/cpu:0'):
  d = cifar10_data(batch_size=batch_size, sess=sess)
  image_batch_tensor, target_batch_tensor = d.build_train_data_tensor(shuffle=True)

## Model
#logits = bn_conv.inference(image_batch_tensor, num_classes=num_classes, is_training=True)
#logits = highway_test.inference(image_batch_tensor, num_classes=num_classes, is_training=True)
from tensorflow.contrib.slim.nets import resnet_v2

with slim.arg_scope(custom_ops.resnet_arg_scope(is_training=True)):
  net, end_points = resnet_v2.resnet_v2_101(image_batch_tensor,
                                              num_classes=num_classes,
                                              global_pool=True)# reduce output to rank 2 (not working)
logits = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=False)

## Losses and Accuracies
classification_loss = slim.losses.softmax_cross_entropy(logits, target_batch_tensor)
total_loss = slim.losses.get_total_loss()
top_1 = tf.nn.in_top_k(logits, tf.argmax(target_batch_tensor, 1), 1)
top_5 = tf.nn.in_top_k(logits, tf.argmax(target_batch_tensor, 1), 5)

top_1_batch_accuracy = tf.reduce_sum(tf.cast(top_1, tf.float32)) * 100.0 / batch_size
top_5_batch_accuracy = tf.reduce_sum(tf.cast(top_5, tf.float32)) * 100.0 / batch_size

## Optimizer
global_step = tf.get_variable('global_step', [],
                  initializer = tf.constant_initializer(0),
                  trainable = False)
learning_rate = tf.placeholder(tf.float32, name="learning_rate")
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
train_op = optimizer.minimize(total_loss, global_step=global_step)

## Summaries
tf.scalar_summary('train/classification_loss', classification_loss)
tf.scalar_summary('train/total_loss', total_loss)
tf.scalar_summary('train/learning_rate', learning_rate)
tf.scalar_summary('train/top_1_batch_acc', top_1_batch_accuracy)
tf.scalar_summary('train/top_5_batch_acc', top_5_batch_accuracy)
summary_op = tf.merge_all_summaries()

## Initialization
saver = tf.train.Saver(max_to_keep=10000000)
summary_writer = tf.train.SummaryWriter(log_dir)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

sess.run(tf.global_variables_initializer())

## Training
for step in range(max_step):
  start_time = time.time()

  lr = 0.1
  if step == 10000 or step*batch_size == 30000:
    lr /= 10
    print("learning rate decrased to ", lr)

  _, summary_str, loss = sess.run([train_op, summary_op, classification_loss], feed_dict={learning_rate: lr})
  duration = time.time() - start_time

  assert not np.isnan(loss), 'Model diverged with loss = NaN'

  if step % 50 == 0:
    examples_per_sec = batch_size / float(duration)
    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
    print(format_str % (datetime.now(), step, loss, examples_per_sec, duration))
    summary_writer.add_summary(summary_str, step)

  if step % 390 == 0 or step == max_step-1:
    print("saving model checkpoint")
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=step)

print("done!")


