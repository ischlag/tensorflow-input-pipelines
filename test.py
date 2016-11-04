import tensorflow as tf
import numpy as np
import threading

r = np.arange(0.0,103.0)
raw_data = np.dstack((r,r,r,r))[0]
raw_target = np.array([[1,0,0]] * 103)

queue1_input_data = tf.placeholder(tf.float32, shape=[20, 4])
queue1_input_target = tf.placeholder(tf.float32, shape=[20, 3])

queue1 = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.float32], shapes=[[4], [3]], shared_name="shared_queue")

enqueue_op = queue1.enqueue_many([queue1_input_data, queue1_input_target])
dequeue_op = queue1.dequeue()

# tensorflow recommendation:
# capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
#data_batch, target_batch = tf.train.shuffle_batch(dequeue_op, batch_size=15, capacity=40, min_after_dequeue=5)
data_batch, target_batch = tf.train.batch(dequeue_op, batch_size=15, capacity=40)

run_options = tf.RunOptions(timeout_in_ms=4000)
sess = tf.Session()

def enqueue(sess):
  under = 0
  max = len(raw_data)
  while True:
    print("starting to write into queue")
    upper = under + 20
    print("try to enqueue ", under, " to ", upper)
    if upper <= max:
      curr_data = raw_data[under:upper]
      curr_target = raw_target[under:upper]
      under = upper
    else:
      rest = upper - max
      curr_data = np.concatenate((raw_data[under:max], raw_data[0:rest]))
      curr_target = np.concatenate((raw_target[under:max], raw_target[0:rest]))
      under = rest

    sess.run(enqueue_op, feed_dict={queue1_input_data: curr_data,
                                    queue1_input_target: curr_target})
    print("added to the queue")
  print("finished enqueueing")


enqueue_thread = threading.Thread(target=enqueue, args=[sess])

enqueue_thread.isDaemon()
enqueue_thread.start()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)



curr_data_batch, curr_target_batch = sess.run([data_batch, target_batch], options=run_options)
print(curr_data_batch)
#print(curr_target_batch)


sess.run(queue1.close(cancel_pending_enqueues=True))

coord.request_stop()
coord.join(threads)
sess.close()