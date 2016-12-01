import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time

from datasets.cifar10 import cifar10_data
from datasets.cifar100 import cifar100_data

from libs import custom_ops
from nets import bn_conv

tf.logging.set_verbosity(tf.logging.INFO)

log_dir = "logs/cifar10/1l_long/"
batch_size = 128
num_classes = 10
epoch_in_steps = int(50000.0/batch_size)
max_step = epoch_in_steps * 15
load_latest_checkpoint = False
step = 0
lrn_rate = 0.1

sess = tf.Session()

## Data
with tf.device('/cpu:0'):
  d = cifar10_data(batch_size=batch_size, sess=sess)
  image_batch_tensor, target_batch_tensor = d.build_train_data_tensor(shuffle=True)

## Model
#logits = bn_conv.inference(image_batch_tensor, num_classes=num_classes, is_training=True)
#logits = highway_test.inference(image_batch_tensor, num_classes=num_classes, is_training=True)
#from tensorflow.contrib.slim.nets import resnet_v2
#with slim.arg_scope(custom_ops.resnet_arg_scope(is_training=True)):
#  net, end_points = resnet_v2.resnet_v2_101(image_batch_tensor,
#                                              num_classes=num_classes,
#                                              global_pool=True)# reduce output to rank 2 (not working)
#logits = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=False)

import nets.resnet
hps = nets.resnet.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.01,
                             num_residual_units=1,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')
model = nets.resnet.ResNet(hps, image_batch_tensor, target_batch_tensor, 'train')
model.build_graph()

## Losses and Accuracies
top_1_correct = tf.nn.in_top_k(model.logits, tf.argmax(target_batch_tensor, 1), 1)
top_5_correct = tf.nn.in_top_k(model.logits, tf.argmax(target_batch_tensor, 1), 5)
top_1_batch_accuracy = tf.reduce_sum(tf.cast(top_1_correct, tf.float32)) * 100.0 / batch_size
top_5_batch_accuracy = tf.reduce_sum(tf.cast(top_5_correct, tf.float32)) * 100.0 / batch_size

## Optimizer

## Summaries
tf.scalar_summary('train/loss', model.cost)
tf.scalar_summary('train/learning_rate', model.lrn_rate)
tf.scalar_summary('train/top_1_batch_acc', top_1_batch_accuracy)
tf.scalar_summary('train/top_5_batch_acc', top_5_batch_accuracy)
summary_op = tf.merge_all_summaries()

## Initialization
saver = tf.train.Saver(max_to_keep=10000000,)
summary_writer = tf.train.SummaryWriter(log_dir)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
sess.run(tf.global_variables_initializer())

## Load Pretrained
if load_latest_checkpoint:
  checkpoint = tf.train.latest_checkpoint(log_dir)
  if checkpoint:
    tf.logging.info("Restoring from checkpoint %s" % checkpoint)
    saver.restore(sess, checkpoint)
    step = sess.run(model.global_step)
  else:
    tf.logging.error("Couldn't find checkpoint to restore from. Exiting.")
    exit()

## Train
tf.logging.info('start training ...')
while not coord.should_stop():
  start_time = time.time()
  (_, summaries, loss, train_step, top_1_acc_val, top_5_acc_val) = sess.run(
    [model.train_op, summary_op, model.cost, model.global_step, top_1_batch_accuracy, top_5_batch_accuracy],
    feed_dict={model.lrn_rate: lrn_rate})

  if train_step < 20000: # 40000
    lrn_rate = 0.1
  elif train_step < 40000: # 60000
    lrn_rate = 0.01
  elif train_step < 60000: # 80000
    lrn_rate = 0.001
  else:
    lrn_rate = 0.0001

  duration = time.time() - start_time

  step += 1
  if step % 50 == 0:
    examples_per_sec = batch_size / float(duration)
    format_str = ('%s: step %4.d, loss: %4.3f, top-1: %5.2f%%, top-5: %5.2f%% (%.1f examples/sec; %.3f sec/batch)')
    tf.logging.info(format_str % (time.strftime("%X"), step, loss,
                                  top_1_acc_val, top_5_acc_val, examples_per_sec, duration))
    summary_writer.add_summary(summaries, step)
    summary_writer.flush()

  if step % 500 == 0:
    tf.logging.info("saving checkpoint")
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=model.global_step)

  if step == 50000:
    exit()

coord.join(threads)

print("done!")
