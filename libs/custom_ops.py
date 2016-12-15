import tensorflow as tf

def log_number_of_params():
  total_parameters = 0
  for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    #tf.logging.info('Shape: %s', shape)
    #tf.logging.info('shape length: %s', len(shape))
    variable_parametes = 1
    for dim in shape:
      #tf.logging.info('dim: %s', dim)
      variable_parametes *= dim.value
    #tf.logging.info('variable params: %s', variable_parametes)
    total_parameters += variable_parametes
  tf.logging.info('Total number of parameters: %s', total_parameters)