##############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  TensorFlow building blocks for models.
# Date:         11.2016
#
#

import tensorflow as tf

def dense(data, 
          n_units,
          phase_train,
          activation,
          scope,
          initializer,
          dropout=True):
	""" Fully-connected network layer."""
	shape = data.get_shape().as_list()
	with tf.variable_scope(scope):
		w = tf.get_variable('dense-weights', 
												[shape[1], n_units],
												initializer=initializer)
		b = tf.get_variable('dense-bias', 
												[n_units], 
												initializer=tf.zeros_initializer)
		dense = activation(tf.matmul(data, w) + b)
		if dropout:
			dense = tf.cond(phase_train, lambda: tf.nn.dropout(dense, 0.5), lambda: dense)
		return dense

def flatten(pre):
	""" Flattens the 2d kernel images into a single vector. Ignore the batch dimensionality."""
	pre_shape = pre.get_shape().as_list()
	flat = tf.reshape(pre, [pre_shape[0], pre_shape[1] * pre_shape[2] * pre_shape[3]])
	return flat

def conv2d(data, 
           n_filters,
           scope,
           initializer,
           k_h=3, k_w=3,
           stride_h=1, stride_w=1,
           bias=True,
           padding='SAME'):
	""" Convolutional layer implementation without an activation function"""
	with tf.variable_scope(scope):
		w = tf.get_variable('conv-weights', 
												[k_h, k_w, data.get_shape()[-1], n_filters],
												initializer=initializer)
		conv = tf.nn.conv2d(data, w, 
												strides=[1, stride_h, stride_w, 1], 
												padding=padding)
		b = tf.get_variable('conv-bias', 
												[n_filters], 
												initializer=tf.zeros_initializer)
		conv = tf.nn.bias_add(conv, b)
		return conv


def batch_norm(x, n_out, phase_train, scope='bn'):
	"""
  Batch normalization on convolutional maps.
  Args:
      x:           Tensor, 4D BHWD input maps
      n_out:       integer, depth of input maps
      phase_train: boolean tf.Varialbe, true indicates training phase
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps

  Note:
    Source is http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  """
	with tf.variable_scope(scope):
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
    										name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
												name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(phase_train,
												mean_var_with_update,
												lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed
