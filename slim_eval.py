
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import math
import time
from datetime import datetime

from datasets.cifar10 import cifar10_data
from datasets.cifar100 import cifar100_data

from libs import custom_ops
from nets import bn_conv

log_dir = "logs/cifar10/3/"
batch_size = 64
num_classes = 10
run_once = False
num_examples = 10000
eval_interval_secs = 10

sess = tf.Session()

## Data
with tf.device('/cpu:0'):
  d = cifar10_data(batch_size=batch_size, sess=sess)
  image_batch_tensor, target_batch_tensor = d.build_test_data_tensor()

## Model
#logits = bn_conv.inference(image_batch_tensor, num_classes=num_classes, is_training=False)
from tensorflow.contrib.slim.nets import resnet_v2
with slim.arg_scope(custom_ops.resnet_arg_scope(is_training=True)):
  net, end_points = resnet_v2.resnet_v2_50(image_batch_tensor,
                                              num_classes=num_classes,
                                              global_pool=True)# reduce output to rank 2 (not working)
logits = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=False)

## Losses and Accuracies
classification_loss = slim.losses.softmax_cross_entropy(logits, target_batch_tensor)
top_1 = tf.nn.in_top_k(logits, tf.argmax(target_batch_tensor, 1), 1)
top_5 = tf.nn.in_top_k(logits, tf.argmax(target_batch_tensor, 1), 5)

top_1_batch_accuracy = tf.reduce_sum(tf.cast(top_1, tf.float32)) * 100.0 / batch_size
top_5_batch_accuracy = tf.reduce_sum(tf.cast(top_5, tf.float32)) * 100.0 / batch_size

## Summaries
tf.scalar_summary('test/classification_loss', classification_loss)
tf.scalar_summary('test/top_1_batch_acc', top_1_batch_accuracy)
tf.scalar_summary('test/top_5_batch_acc', top_5_batch_accuracy)
summary_op = tf.merge_all_summaries()

## Initialization
saver = tf.train.Saver()
summary_writer = tf.train.SummaryWriter(log_dir)

print("ready")
def _eval_model_checkpoint(sess, model_checkpoint_path, saver, summary_writer, top_1, top_5, summary_op):
  # load latest checkpoint file
  try:
    # Restores from checkpoint with relative path.
    saver.restore(sess, model_checkpoint_path)

    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/imagenet_train/model.ckpt-0,
    # extract global_step from it.
    global_step = model_checkpoint_path.split('/')[-1].split('-')[-1]
    print('Succesfully loaded model from %s at step=%s.' % (model_checkpoint_path, global_step))
  except Exception as e:
    print('No checkpoint file found in ', log_dir, " -> ", e)
    return

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord, sess=sess)
  try:
    num_iter = int(math.ceil(float(num_examples) / batch_size))
    # Counts the number of correct predictions.
    count_top_1 = 0.0
    count_top_5 = 0.0
    total_sample_count = num_iter * batch_size
    step = 0

    print('%s: starting evaluation on (%s).' % (datetime.now(), "test"))
    start_time = time.time()
    while step < num_iter and not coord.should_stop():
      top_1_val, top_5_val = sess.run([top_1, top_5])
      count_top_1 += np.sum(top_1_val)
      count_top_5 += np.sum(top_5_val)
      step += 1
      if step % 20 == 0:
        duration = time.time() - start_time
        sec_per_batch = duration / 20.0
        examples_per_sec = batch_size / sec_per_batch
        print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
              'sec/batch)' % (datetime.now(), step, num_iter,
                              examples_per_sec, sec_per_batch))
        start_time = time.time()

    # Compute precision @ 1.
    precision_at_1 = count_top_1 / total_sample_count
    recall_at_5 = count_top_5 / total_sample_count
    print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
          (datetime.now(), precision_at_1, recall_at_5, total_sample_count))

    summary = tf.Summary()
    summary.ParseFromString(sess.run(summary_op))
    summary.value.add(tag='test/Precision @ 1', simple_value=precision_at_1)
    summary.value.add(tag='test/Recall @ 5', simple_value=recall_at_5)
    summary_writer.add_summary(summary, global_step)

  except Exception as e:  # pylint: disable=broad-except
    coord.request_stop(e)

  coord.request_stop()
  coord.join(threads, stop_grace_period_secs=10)


if run_once:
  ckpt = tf.train.get_checkpoint_state(log_dir)
  if ckpt and ckpt.model_checkpoint_path:
    _eval_model_checkpoint(sess, ckpt.model_checkpoint_path, saver, summary_writer, top_1, top_5, summary_op)
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
          _eval_model_checkpoint(sess, path, saver, summary_writer, top_1, top_5, summary_op)
    else:
      print('No checkpoint file found')
    time.sleep(eval_interval_secs)

print("done!")
























