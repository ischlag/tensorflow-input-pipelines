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

log_dir = "logs/cifar10/1l_long"
batch_size = 128
num_classes = 10
epoch_size = 10000.0
num_iter = int(math.ceil(epoch_size/batch_size))
load_latest_checkpoint = False

eval_interval_secs = 3
run_once = False

tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.Session()

## Data
with tf.device('/cpu:0'):
  d = cifar10_data(batch_size=batch_size, sess=sess)
  image_batch_tensor, target_batch_tensor = d.build_test_data_tensor(shuffle=False)

## Model
#logits = bn_conv.inference(image_batch_tensor, num_classes=num_classes, is_training=True)
#from tensorflow.contrib.slim.nets import resnet_v2
#with slim.arg_scope(custom_ops.resnet_arg_scope(is_training=True)):
#  net, end_points = resnet_v2.resnet_v2_101(image_batch_tensor,
#                                              num_classes=num_classes,
#                                              global_pool=True)# reduce output to rank 2 (not working)
#logits = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=False)
import nets.resnet
hps = nets.resnet.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=None,
                             lrn_rate=None,
                             num_residual_units=1,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')
model = nets.resnet.ResNet(hps, image_batch_tensor, target_batch_tensor, 'eval')
model.build_graph()

## Losses and Accuracies
top_1_correct = tf.nn.in_top_k(model.logits, tf.argmax(target_batch_tensor, 1), 1)
top_5_correct = tf.nn.in_top_k(model.logits, tf.argmax(target_batch_tensor, 1), 5)
top_1_batch_accuracy = tf.reduce_sum(tf.cast(top_1_correct, tf.float32)) * 100.0 / batch_size
top_5_batch_accuracy = tf.reduce_sum(tf.cast(top_5_correct, tf.float32)) * 100.0 / batch_size

## Optimizer

## Summaries
# Don't!

## Initialization
saver = tf.train.Saver(max_to_keep=10000000)
summary_writer = tf.train.SummaryWriter(log_dir)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
sess.run(tf.global_variables_initializer())


def _eval_model_checkpoint(model_checkpoint_path):
  if model_checkpoint_path:
    tf.logging.info("Restoring from checkpoint %s" % model_checkpoint_path)
    saver.restore(sess, model_checkpoint_path)
  else:
    tf.logging.error("Couldn't find checkpoint to restore from. Exiting.")
    return

  # Counts the number of correct predictions.
  count_top_1 = 0.0
  count_top_5 = 0.0
  count_avg_loss = 0.0
  total_sample_count = num_iter * batch_size
  step = 0
  global_step = model_checkpoint_path.split('/')[-1].split('-')[-1]

  tf.logging.info('%s: starting evaluation.' % (datetime.now()))
  start_time = time.time()
  while step < num_iter and not coord.should_stop():
    top_1_val, top_5_val, loss_value = sess.run([top_1_correct, top_5_correct, model.cost])
    count_top_1 += np.sum(top_1_val)
    count_top_5 += np.sum(top_5_val)
    count_avg_loss += np.mean(loss_value)
    step += 1
    if step % 40 == 0:
      duration = time.time() - start_time
      sec_per_batch = duration / 20.0
      examples_per_sec = batch_size / sec_per_batch
      tf.logging.info('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
            'sec/batch)' % (datetime.now(), step, num_iter,
                            examples_per_sec, sec_per_batch))
      start_time = time.time()

  # compute test set accuracy
  top_1_accuracy = count_top_1 / total_sample_count
  top_5_accuracy = count_top_5 / total_sample_count
  avg_loss = count_avg_loss / total_sample_count
  tf.logging.info('%s: top_1_acc=%.4f, top_5_acc=%.4f avg_loss=%.7f [%d examples]' %
        (datetime.now(), top_1_accuracy, top_5_accuracy, count_avg_loss, total_sample_count))

  top_1_summary = tf.Summary(value=[tf.Summary.Value(tag="test/top_1_accuracy", simple_value=top_1_accuracy)])
  top_5_summary = tf.Summary(value=[tf.Summary.Value(tag="test/top_5_accuracy", simple_value=top_5_accuracy)])
  avg_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="test/avg_loss", simple_value=avg_loss)])
  summary_writer.add_summary(top_1_summary, global_step)
  summary_writer.add_summary(top_5_summary, global_step)
  summary_writer.add_summary(avg_loss_summary, global_step)
  summary_writer.flush()

## Eval
if run_once:
  ckpt = tf.train.get_checkpoint_state(log_dir)
  if ckpt and ckpt.model_checkpoint_path:
    _eval_model_checkpoint(ckpt.model_checkpoint_path)
  else:
    tf.logging.error('No checkpoint file found')
    exit()

else:
  done = []
  while True:
    tf.logging.info("checking for new models in %s ... " % log_dir)
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
      for path in ckpt.all_model_checkpoint_paths:
        if not path in done:
          done.append(path)
          _eval_model_checkpoint(path)
    else:
      tf.logging.error('No checkpoint file found')
    time.sleep(eval_interval_secs)

print("done!")
