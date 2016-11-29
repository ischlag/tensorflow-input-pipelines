import tensorflow as tf
import tensorflow.contrib.slim as slim


def resblock(inputs,
             num_filters,
             weighted_skip=False,
             kernel_size=(3, 3),
             stride=1,
             mid_stride=1,
             filters_ratio=4,
             padding='SAME',
             activation=tf.nn.relu,
             stddev=None,
             bias=0.0,
             weight_decay=0,
             batch_norm_params=None,
             is_training=True,
             trainable=True,
             restore=True,
             scope=None,
             reuse=None
             ):
  with tf.variable_op_scope([inputs], scope, 'resblock', reuse=reuse):
    with slim.arg_scope([slim.conv2d], num_filters_out=num_filters,
                          kernel_size=kernel_size, stride=stride,
                          padding=padding, activation=activation,
                          stddev=stddev, bias=bias,
                          weight_decay=weight_decay,
                          batch_norm_params=batch_norm_params,
                          is_training=is_training, trainable=trainable,
                          restore=restore, scope=scope, reuse=reuse):
      h = slim.conv2d(inputs, kernel_size=(1, 1))
      h = slim.conv2d(h, stride=mid_stride)
      h = slim.conv2d(h, num_filters_out=filters_ratio*num_filters,
                 kernel_size=(1, 1), activation=None)
      if weighted_skip or mid_stride > 1:
        x = slim.conv2d(inputs, num_filters_out=filters_ratio*num_filters,
                   kernel_size=(1, 1), stride=mid_stride, activation=None)
      else:
        x = inputs
      outputs = activation(h + x)
    return outputs


def hwblock(inputs,
            num_filters,
            weighted_skip=False,
            biases_initializer=tf.constant_initializer(-1.0),
            kernel_size=(3, 3),
            stride=1,
            mid_stride=1,
            filters_ratio=4,
            padding='SAME',
            activation_fn=tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            weights_regularizer=None,
            trainable=True,
            scope=None,
            reuse=None,
            is_training=True
            ):
  with tf.variable_op_scope([inputs], scope, 'hwblock', reuse=reuse):
    with slim.arg_scope([slim.conv2d],
                        num_outputs=num_filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        activation_fn=activation_fn,
                        weights_initializer=weights_initializer,
                        weights_regularizer=weights_regularizer,
                        trainable=trainable,
                        scope=scope,
                        reuse=reuse):
      h = slim.conv2d(inputs, kernel_size=(1, 1))
      h = slim.batch_norm(h, decay=0.9, epsilon=0.00005, is_training=is_training)
      h = slim.conv2d(h, stride=mid_stride)
      h = slim.batch_norm(h, decay=0.9, epsilon=0.00005, is_training=is_training)
      h = slim.conv2d(h, num_outputs=filters_ratio*num_filters, kernel_size=(1, 1), activation_fn=None)
      h = slim.batch_norm(h, decay=0.9, epsilon=0.00005, is_training=is_training)

      t = slim.conv2d(inputs, kernel_size=(1, 1))
      t = slim.batch_norm(t, decay=0.9, epsilon=0.00005, is_training=is_training)
      t = slim.conv2d(t, stride=mid_stride)
      t = slim.batch_norm(t, decay=0.9, epsilon=0.00005, is_training=is_training)
      t = slim.conv2d(t, num_outputs=filters_ratio * num_filters,
                      kernel_size=(1, 1), activation_fn=tf.sigmoid, biases_initializer=biases_initializer)
      t = slim.batch_norm(t, decay=0.9, epsilon=0.00005, is_training=is_training)
      if weighted_skip or mid_stride > 1:
        x = slim.conv2d(inputs, num_outputs=filters_ratio*num_filters,
                   kernel_size=(1, 1), stride=mid_stride, activation_fn=None)
        x = slim.batch_norm(x, decay=0.9, epsilon=0.00005, is_training=is_training)
      else:
        x = inputs
      outputs = (h - x) * t + x
    return outputs

def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.9,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  """Defines the default ResNet arg scope.
  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.
  Args:
    is_training: Whether or not we are training the parameters in the batch
      normalization layers of the model.
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
    'is_training': is_training,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope(
          [slim.conv2d],
          weights_regularizer=slim.l2_regularizer(weight_decay),
          weights_initializer=slim.variance_scaling_initializer(), #mode='FAN_OUT'
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc