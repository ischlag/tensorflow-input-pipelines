import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
import math
from datetime import datetime

from datasets.cifar10 import cifar10_data
from datasets.cifar100 import cifar100_data

from libs import custom_ops
from nets import bn_conv
from nets import highway_test

log_dir = "logs/cifar10/1/"
eval_dir = "logs/cifar10/1_eval/"
batch_size = 64
num_classes = 10
epoch_in_steps = int(10000.0/batch_size)
max_step = epoch_in_steps * 15
load_latest_checkpoint = True
eval_interval_secs = 10
run_once = False

sess = tf.Session()

## Data
with tf.device('/cpu:0'):
  d = cifar10_data(batch_size=batch_size, sess=sess)
  image_batch_tensor, target_batch_tensor = d.build_test_data_tensor(shuffle=False)

## Model
logits = bn_conv.inference(image_batch_tensor, num_classes=num_classes, is_training=True)
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


## Summaries
tf.scalar_summary('test/loss', loss)
tf.scalar_summary('test/top_1_batch_acc', top_1_batch_accuracy)
tf.scalar_summary('test/top_5_batch_acc', top_5_batch_accuracy)
summary_op = tf.merge_all_summaries()

## Initialization
saver = tf.train.Saver(max_to_keep=10000000,)
summary_writer = tf.train.SummaryWriter(eval_dir)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

sess.run(tf.global_variables_initializer())

def _eval_model_checkpoint(model_checkpoint_path):
  if model_checkpoint_path:
      print("Restoring from checkpoint", model_checkpoint_path)
      saver.restore(sess, model_checkpoint_path)
  else:
    print("Couldn't find checkpoint to restore from. Exiting.")
    return

  num_iter = int(math.ceil(float(50000.0) / batch_size))
  # Counts the number of correct predictions.
  count_top_1 = 0.0
  count_top_5 = 0.0
  total_sample_count = num_iter * batch_size
  step = 0
  global_step = model_checkpoint_path.split('/')[-1].split('-')[-1]

  print('%s: starting evaluation.' % (datetime.now()))
  start_time = time.time()
  while step < num_iter and not coord.should_stop():
    top_1_val, top_5_val, summary_value = sess.run([top_1_correct, top_5_correct, summary_op])
    count_top_1 += np.sum(top_1_val)
    count_top_5 += np.sum(top_5_val)
    step += 1
    if step % 40 == 0:
      duration = time.time() - start_time
      sec_per_batch = duration / 20.0
      examples_per_sec = batch_size / sec_per_batch
      print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
            'sec/batch)' % (datetime.now(), step, num_iter,
                            examples_per_sec, sec_per_batch))
      summary_writer.add_summary(summary_value, global_step)
      start_time = time.time()

  # compute test set accuracy
  top_1_accuracy = count_top_1 / total_sample_count
  top_5_accuracy = count_top_5 / total_sample_count
  print('%s: top_1_acc=%.4f, top_5_acc=%.4f [%d examples]' %
        (datetime.now(), top_1_accuracy, top_5_accuracy, total_sample_count))

  top_1_summary = tf.Summary(value=[tf.Summary.Value(tag="test/top_1_accuracy", simple_value=top_1_accuracy)])
  top_5_summary = tf.Summary(value=[tf.Summary.Value(tag="test/top_5_accuracy", simple_value=top_5_accuracy)])
  summary_writer.add_summary(top_1_summary, global_step)
  summary_writer.add_summary(top_5_summary, global_step)

# Eval
if run_once:
  ckpt = tf.train.get_checkpoint_state(log_dir)
  if ckpt and ckpt.model_checkpoint_path:
    _eval_model_checkpoint(ckpt.model_checkpoint_path)
  else:
    print('No checkpoint file found')
    exit()

else:
  done = []
  while True:
    print("checking for new models ...")
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
      for path in ckpt.all_model_checkpoint_paths:
        if not path in done:
          done.append(path)
          _eval_model_checkpoint(path)
    else:
      print('No checkpoint file found')
    time.sleep(eval_interval_secs)


print("done!")
