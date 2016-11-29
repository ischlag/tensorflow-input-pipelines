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

log_dir = "logs/cifar10/1/"
batch_size = 64
num_classes = 10
epoch_in_steps = int(50000.0/batch_size)
max_step = epoch_in_steps * 15
load_latest_checkpoint = False

sess = tf.Session()

## Data
with tf.device('/cpu:0'):
  d = cifar10_data(batch_size=batch_size, sess=sess)
  image_batch_tensor, target_batch_tensor = d.build_train_data_tensor(shuffle=True)

## Model
logits = bn_conv.inference(image_batch_tensor, num_classes=num_classes, is_training=True)
#logits = highway_test.inference(image_batch_tensor, num_classes=num_classes, is_training=True)
#from tensorflow.contrib.slim.nets import resnet_v2
#with slim.arg_scope(custom_ops.resnet_arg_scope(is_training=True)):
#  net, end_points = resnet_v2.resnet_v2_101(image_batch_tensor,
#                                              num_classes=num_classes,
#                                              global_pool=True)# reduce output to rank 2 (not working)
#logits = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=False)

## Losses and Accuracies
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                        tf.cast(target_batch_tensor, tf.float32),
                                                        name="cross-entropy")
loss = tf.reduce_mean(cross_entropy, name='loss')

top_1_correct = tf.nn.in_top_k(logits, tf.argmax(target_batch_tensor, 1), 1)
top_5_correct = tf.nn.in_top_k(logits, tf.argmax(target_batch_tensor, 1), 5)

top_1_batch_accuracy = tf.reduce_sum(tf.cast(top_1_correct, tf.float32)) * 100.0 / batch_size
top_5_batch_accuracy = tf.reduce_sum(tf.cast(top_5_correct, tf.float32)) * 100.0 / batch_size

## Optimizer
global_step = tf.Variable(0, name='global_step', trainable=False)
learning_rate = tf.placeholder(tf.float32, name="learning_rate")
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
train_op = optimizer.minimize(loss, global_step=global_step)

## Summaries
tf.scalar_summary('train/loss', loss)
tf.scalar_summary('train/learning_rate', learning_rate)
tf.scalar_summary('train/top_1_batch_acc', top_1_batch_accuracy)
tf.scalar_summary('train/top_5_batch_acc', top_5_batch_accuracy)
summary_op = tf.merge_all_summaries()

## Initialization
saver = tf.train.Saver(max_to_keep=10000000,)
summary_writer = tf.train.SummaryWriter(log_dir)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

sess.run(tf.global_variables_initializer())

if load_latest_checkpoint:
  checkpoint = tf.train.latest_checkpoint(log_dir)
  if checkpoint:
      print("Restoring from checkpoint", checkpoint)
      saver.restore(sess, checkpoint)
  else:
    print("Couldn't find checkpoint to restore from. Exiting.")
    exit()

## Training
epoch_count = 0
lr = 0.01
for step in range(max_step):
  start_time = time.time()

  if step % (epoch_in_steps*10) == 0 and step > 100:
    lr /= 10
    print("learning rate decrased to ", lr)

  if step % epoch_in_steps == 0:
    epoch_count += 1
    print("epoch: ", epoch_count)

  _, summary_str, loss_val = sess.run([train_op, summary_op, loss], feed_dict={learning_rate: lr})
  duration = time.time() - start_time

  assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

  if step % 50 == 0:
    examples_per_sec = batch_size / float(duration)
    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
    print(format_str % (datetime.now(), step, loss_val, examples_per_sec, duration))
    summary_writer.add_summary(summary_str, step)

  if step % 390 == 0 or step == max_step-1:
    print("saving model checkpoint")
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=global_step)

print("done!")
