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

log_dir = "logs/cifar10/6stages_2res_0final_RandNorm_sgd/"
eval_dir = log_dir
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
import nets.resnet_uniform
hps = nets.resnet_uniform.HParams(batch_size=batch_size,
                          num_classes=num_classes,
                          min_lrn_rate=None,
                          lrn_rate=None,
                          num_residual_units=2,
                          use_bottleneck=False,
                          weight_decay_rate=0.0002,
                          relu_leakiness=0.1,
                          optimizer='mom')
model = nets.resnet_uniform.ResNet(hps, image_batch_tensor, target_batch_tensor, 'eval')
model.build_graph()

## Losses and Accuracies

## Optimizer

## Summaries

## Initialization
saver = tf.train.Saver(max_to_keep=10000000)
summary_writer = tf.train.SummaryWriter(eval_dir)
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
  total_loss = 0.0
  total_sample_count = num_iter * batch_size
  step = 0
  global_step = model_checkpoint_path.split('/')[-1].split('-')[-1]

  tf.logging.info('%s: starting evaluation.' % (datetime.now()))
  start_time = time.time()
  correct_prediction, total_prediction = 0, 0
  while step < num_iter and not coord.should_stop():
    loss_value, predictions, truth = sess.run([model.cost, model.predictions, model.labels])

    total_loss += np.sum(loss_value)
    truth = np.argmax(truth, axis=1)
    predictions = np.argmax(predictions, axis=1)
    correct_prediction += np.sum(truth == predictions)
    total_prediction += predictions.shape[0]

    step += 1
    if step % 200 == 0:
      duration = time.time() - start_time
      sec_per_batch = duration / 20.0
      examples_per_sec = batch_size / sec_per_batch
      tf.logging.info('[%d batches out of %d] (%.1f examples/sec; %.3f'
            'sec/batch)' % (step, num_iter, examples_per_sec, sec_per_batch))
      start_time = time.time()

  # compute test set accuracy
  accuracy = correct_prediction * 100.0 / total_prediction
  avg_loss = total_loss / total_sample_count

  tf.logging.info('%s: top_1_acc: %6.3f%%, avg_loss: %.7f [%d examples]' %
        (global_step, accuracy, avg_loss, total_sample_count))

  accuracy_sum = tf.Summary(value=[tf.Summary.Value(tag="test/accuracy", simple_value=accuracy)])
  avg_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="test/avg_loss", simple_value=avg_loss)])
  summary_writer.add_summary(accuracy_sum, global_step)
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
