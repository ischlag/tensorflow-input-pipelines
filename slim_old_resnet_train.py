import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
from collections import deque

from datasets.cifar10 import cifar10_data
from datasets.cifar100 import cifar100_data

from libs import components
from libs import custom_ops
from nets import bn_conv

tf.logging.set_verbosity(tf.logging.INFO)

log_dir = "logs/cifar10/old_small/"
ckpt_dir = log_dir # "logs/cifar10/wrn_1/"
batch_size = 64
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
  image_batch_tensor, target_batch_tensor = d.build_train_data_tensor(shuffle=True, augmentation=True)

## Model
#logits = bn_conv.inference(image_batch_tensor, num_classes=num_classes, is_training=True)
#logits = highway_test.inference(image_batch_tensor, num_classes=num_classes, is_training=True)
#from tensorflow.contrib.slim.nets import resnet_v2
#with slim.arg_scope(custom_ops.resnet_arg_scope(is_training=True)):
#  net, end_points = resnet_v2.resnet_v2_101(image_batch_tensor,
#                                                num_classes=num_classes,
#                                              global_pool=True)# reduce output to rank 2 (not working)
#logits = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=False)

#import nets.resnet
import nets.resnet_old_reference
hps = nets.resnet_old_reference.HParams(batch_size=batch_size,
                          num_classes=num_classes,
                          min_lrn_rate=0.0001,
                          lrn_rate=0.1,
                          num_residual_units=4,
                          use_bottleneck=False,
                          weight_decay_rate=0.0002,
                          relu_leakiness=0.1,
                          optimizer='mom')
model = nets.resnet_old_reference.ResNet(hps, image_batch_tensor, target_batch_tensor, 'train')
model.build_graph()

## Losses and Accuracies
avg_loss_queue = deque(epoch_in_steps * [0])
avg_top1_queue = deque(epoch_in_steps * [0])

## Optimizer

## Summaries
summary_op = model.summaries

## Initialization
saver = tf.train.Saver(max_to_keep=10000000)
summary_writer = tf.train.SummaryWriter(log_dir, graph=sess.graph)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
sess.run(tf.global_variables_initializer())

## Load Pretrained
if load_latest_checkpoint:
  checkpoint = tf.train.latest_checkpoint(ckpt_dir)
  if checkpoint:
    tf.logging.info("Restoring from checkpoint %s" % checkpoint)
    saver.restore(sess, checkpoint)
    step = sess.run(model.global_step)
  else:
    tf.logging.error("Couldn't find checkpoint to restore from. Exiting.")
    exit()

## Train
tf.logging.info('start training ...')
total_start_time = time.time()
while not coord.should_stop():
  start_time = time.time()
  correct_prediction, total_prediction = 0, 0
  (_, summaries, loss, train_step, predictions, truth) = sess.run(
    [model.train_op, summary_op, model.cost, model.global_step, model.predictions, model.labels],
    feed_dict={model.lrn_rate: lrn_rate})

  if train_step < 20000: # 15000: # 40000
    lrn_rate = 0.1
  elif train_step < 40000: #30000: # 60000
    lrn_rate = 0.01
  elif train_step < 50000: # # 80000
    lrn_rate = 0.001
  else:
    lrn_rate = 0.0001

  duration = time.time() - start_time
  truth = np.argmax(truth, axis=1)
  predictions = np.argmax(predictions, axis=1)
  accuracy = np.sum(truth == predictions) * 100 / batch_size
  avg_accuracy = components.push_into_queue(accuracy, avg_top1_queue, "train/avg_accuracy", train_step, summary_writer)
  avg_loss = components.push_into_queue(loss, avg_loss_queue, "train/avg_loss", train_step, summary_writer)

  if step % 100 == 0:
    total_duration = (time.time() - total_start_time) / 60.0
    examples_per_sec = batch_size / float(duration)
    accuracy_sum = tf.Summary(value=[tf.Summary.Value(tag="train/accuracy", simple_value=accuracy)])

    if step == 500:
      format_str = ('%4.2fmin, step %4.d, lr: %.4f, loss: %4.3f, top-1: %5.2f%% (%.1f examples/sec; %.3f sec/batch)')
      tf.logging.info(format_str % (total_duration, step, lrn_rate, loss, accuracy, examples_per_sec, duration))
    else:
      format_str = ('%4.2fmin, step %4.d, lr: %.4f, loss: %4.3f (%4.3f), top-1: %5.2f%% (%5.2f%%)')
      tf.logging.info(format_str % (total_duration, step, lrn_rate, loss, avg_loss, accuracy, avg_accuracy))

    summary_writer.add_summary(accuracy_sum, train_step)
    summary_writer.add_summary(summaries, train_step)
    summary_writer.flush()

  if step % 1000 == 0:
    tf.logging.info("saving checkpoint")
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=model.global_step)

  step += 1

coord.join(threads)

print("done!")
