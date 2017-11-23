from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from general_queue_runner import GeneralQueueRunner

a = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], dtype=np.int32)
test_samples = np.tile(a, [5,1])
test_samples = np.transpose(test_samples)
idx = 0
batch_size = 4

b = [[0, 1, 2, 3],
     [0, 1, 2, 3, 4],
     [0, 1, 2, 3, 4, 5],
     [0, 1, 2, 3, 4, 6],
     ]

def read_func():
    global idx, test_samples
    #print("\nAdded to the queue\n")
    if idx >= test_samples.shape[0]:
        idx = 0

    sample = test_samples[idx]
    idx += 1

    return np.reshape(sample,-1).tobytes()


def read_func_0():
    global idx, b
    #print("\nAdded to the queue\n")
    if idx >= len(b):
        idx = 0

    sample = b[idx]
    idx += 1

    return np.reshape(sample,-1).tobytes()

serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
feature_infos={
    'x': tf.FixedLenFeature([], tf.string)
}

# decoder
tf_example = tf.parse_single_example(serialized_tf_example, feature_infos)
# print tf_example['x'].dtype

queue = tf.FIFOQueue(capacity=20, dtypes=[tf.string], name='fifo_queue')
size = queue.size()
enqueue_op = queue.enqueue(tf_example['x'], name="enqueue")
value = queue.dequeue()

def test_fixed_shape(value):
    qr = GeneralQueueRunner(
        queue=queue,
        enqueue_ops=[enqueue_op, enqueue_op],
        feed_dict_funcs=[read_func, read_func],
        feed_tensors=[tf_example['x'], tf_example['x']]
    )
    tf.train.add_queue_runner(qr)

    decode_x = tf.decode_raw(value, tf.int32)
    decode_x = tf.reshape(decode_x, [5])
    data_batch = tf.train.shuffle_batch([decode_x], batch_size=batch_size, capacity=24, num_threads=1, min_after_dequeue=6)

    sess =  tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)
    for _ in range(0, 20):
        run_options = tf.RunOptions(timeout_in_ms=4000)
        curr_data_batch = sess.run([data_batch], options=run_options)
        print(curr_data_batch)
        print("\n")

    coord.request_stop()
    coord.join(threads)
    sess.close()

def test_dynamic_shape(value):
    qr = GeneralQueueRunner(
        queue=queue,
        enqueue_ops=[enqueue_op],
        feed_dict_funcs=[read_func_0],
        feed_tensors=[tf_example['x']]
    )

    tf.train.add_queue_runner(qr)
    value = tf.reshape(value,[])

    data_batch = tf.train.batch([value], batch_size=batch_size, capacity=24, num_threads=1)

    sess =  tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)
    for _ in range(0, 2):
        run_options = tf.RunOptions(timeout_in_ms=4000)
        curr_data_batch = sess.run(data_batch, options=run_options)

        for v in curr_data_batch:
            r = np.fromstring(v, dtype=np.int64)
            print(r)
        print("\n")

    coord.request_stop()
    coord.join(threads)
    sess.close()

FIXED_TEST = True
if FIXED_TEST: test_fixed_shape(value)
else: test_dynamic_shape(value)
