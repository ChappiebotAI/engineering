import tensorflow as tf
import random

slim = tf.contrib.slim


class TRexCNN(object):
    def __init__(self, discount=0.95, using_cuda=True):
        self.reuse = False
        self.discount = discount
        self.dev = "/cpu:0"
        if using_cuda:
            self.dev = "/gpu:0"
        pass

    def build_predict_op(self, inputs):
        current_state = inputs[0]
        epsilon = inputs[1]

        with tf.device(self.dev):
            q_value = self._calc_q_value(current_state, reuse=self.reuse)
            coin = tf.random_uniform((), minval=0., maxval=1.)
            cond = tf.less(coin, epsilon)
            action = tf.cond(cond, 
                            lambda: tf.to_int64(tf.floor(tf.random_uniform((), minval=0., maxval=1.)*2)),
                            lambda: tf.to_int64(tf.argmax(q_value, axis=1)))
        return q_value, action

    def build_train_op(self, inputs):
        current_state = inputs[0]
        action = inputs[1]
        reward = inputs[2]
        next_state = inputs[3]
        terminal = inputs[4]

        with tf.device(self.dev):
            current_q_values = self._calc_q_value(current_state, reuse=self.reuse)
            idx = tf.range(0, current_q_values.shape[0]) * current_q_values.shape[1] + action
            idx = tf.to_int64(idx)
            current_q_value = tf.gather(tf.reshape(current_q_values, [-1]), idx)

            next_q_values = self._calc_q_value(next_state, reuse=self.reuse)
            max_action = tf.argmax(next_q_values, axis=1)
            #next_q_value = next_q_values[:,max_action]

            idx = tf.to_int64(tf.range(0, next_q_values.shape[0]) * next_q_values.shape[1]) + tf.to_int64(max_action)
            next_q_value = tf.gather(tf.reshape(next_q_values, [-1]), idx)

            # Caculating loss function
            tt = reward + tf.to_float(1 - terminal) * next_q_value * self.discount
            tq = tf.square(current_q_value - tt)
            loss = tf.reduce_mean(tq)

            # Build optimizer ops
            global_step = tf.contrib.framework.get_or_create_global_step()
            lrn_rate = 1e-6
            optimizer = tf.train.AdamOptimizer(lrn_rate)
            grads = optimizer.compute_gradients(loss)
            apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

            with tf.control_dependencies([apply_gradient_op]):
                loss = tf.identity(loss)

        return apply_gradient_op, global_step, loss

    def _calc_q_value(self, x, reuse=False):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(0.00005),
                      biases_regularizer=None):
            with tf.variable_scope('t_rex', reuse=reuse):
                x = slim.conv2d(x, 32, 8, 4, 
                                activation_fn=tf.nn.relu, 
                                scope="conv1", padding="SAME")
                x = slim.max_pool2d(x, 1, 2, scope="max_pool1", padding="SAME")
                x = slim.conv2d(x, 64, 4, 2, 
                                activation_fn=tf.nn.relu, 
                                scope="conv2", padding="SAME")
                x = slim.conv2d(x, 64, 3, 1, 
                                activation_fn=tf.nn.relu, 
                                scope="conv3", padding="SAME")
                pool_shape = x.get_shape().as_list()
                flat_size = pool_shape[1]*pool_shape[2]*pool_shape[3]
                x = tf.reshape(x, [-1, flat_size])
                x = slim.fully_connected(x, 256, 
                                    activation_fn=tf.nn.relu, scope='fc1')
                x = slim.fully_connected(x, 2, 
                                    activation_fn=None, scope='fc2')

        if not self.reuse: 
            self.reuse = True
        return x