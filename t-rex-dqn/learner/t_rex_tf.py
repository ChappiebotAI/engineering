import tensorflow as tf
import threading
from model import TRexCNN
import time
import numpy as np
import os
import random


class TRexGaming(object):
    def __init__(
            self, 
            prepared_queue, 
            mode="learn",
            discount=0.95, 
            epsilon=0.075,
            batch_size=32, 
            using_cuda=True,
            replay_size=50000):
        self.prepared_queue = prepared_queue
        self.mode = mode
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.using_cuda = using_cuda
        self.epsilon = epsilon
        self.model = TRexCNN(discount=discount, using_cuda=using_cuda)
        self.dev = "/cpu:0"
        if using_cuda:
            self.dev = "/gpu:0"

    def create_input_tensors(self, batch_size=32, for_training=True):
        curr_state_input = tf.placeholder(tf.float32, shape=[batch_size, 50, 200, 4])
        if not for_training:
            return curr_state_input
        action_input = tf.placeholder(tf.int32, shape=[batch_size])
        reward_input = tf.placeholder(tf.float32, shape=[batch_size])
        next_state_input = tf.placeholder(tf.float32, shape=[batch_size, 50, 200, 4])
        terminal = tf.placeholder(tf.float32, shape=[batch_size])
        return curr_state_input, action_input, reward_input, next_state_input, terminal

    @staticmethod
    def train_thread(sess, train_op, global_step, loss, inputs,
                     prepared_queue, batch_size, max_steps,
                     save_steps, training_saver, dev, replay_size):
        step = 0
        wait_full_replay_mem = True
       
        while(True):
            if step==0 : time.sleep(1)
            
            curr_state = []
            action = []
            reward = []
            next_state = []
            terminal = []
                
            if len(prepared_queue) <= replay_size and wait_full_replay_mem:
                print("Relay memory:%d" % len(prepared_queue))
                if len(prepared_queue) == replay_size:
                    wait_full_replay_mem = False
                continue

            if len(prepared_queue) < batch_size:
                continue

            items = random.sample(prepared_queue, batch_size)
            for item in items:
                curr_state.append(item[0])
                action.append(item[1])
                reward.append(item[2])
                next_state.append(item[3])
                terminal.append(item[4])
            
                # for j in sorted(indexes, reverse=True):
                #     del prepared_queue[j]
            time.sleep(0.001)    
            curr_state = np.stack(curr_state, axis=0)
            action = np.stack(action, axis=0)
            reward = np.stack(reward, axis=0)
            next_state = np.stack(next_state, axis=0)
            terminal = np.stack(terminal, axis = 0)
            _, ret_loss, ret_global_step = sess.run([train_op, loss, global_step],
                            feed_dict={inputs[0]:curr_state, inputs[1]:action, 
                                       inputs[2]:reward, inputs[3]:next_state,
                                       inputs[4]:terminal})
            #train_op.run(feed_dict={inputs[0]:curr_state, inputs[1]:action, inputs[2]:reward, inputs[3]:next_state},
            #             session=sess)
            step += 1
            if step % 50 == 0:
                print("Step %d: %0.3f" % (ret_global_step, ret_loss))
            if step >= max_steps:
                break
            if step % save_steps == 0:
                training_saver.save(sess, 't-rex-DQN', global_step=ret_global_step)
        sess.close()

    def running(self, max_steps=20000000, save_steps=5000, checkpoint=None):
        with tf.Graph().as_default(), tf.device("/cpu:0"):
            # Build tensorflow ops for predicting and training processes
            inputs = self.create_input_tensors(batch_size=self.batch_size)
            self.curr_state_input = self.create_input_tensors(batch_size=1, for_training=False)

            epsilon = tf.constant(0.0)
            if self.mode == "learn":
                train_op, global_step, loss = self.model.build_train_op(inputs)
                epsilon = tf.train.exponential_decay(
                                    0.075,
                                    global_step,
                                    100000,
                                    0.5,
                                    staircase=True)
            self.pred_q_value, self.pred_action = \
                self.model.build_predict_op([self.curr_state_input, epsilon])

            config = tf.ConfigProto(allow_soft_placement = True)
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            if checkpoint:
                restore_saver = tf.train.Saver(
                                    var_list=tf.trainable_variables() + [global_step,])
                restore_saver.restore(self.sess, checkpoint) 

            if self.mode == "learn":
                training_saver = tf.train.Saver(
                    var_list=tf.trainable_variables() + [global_step,], max_to_keep=10)
                train_thread = threading.Thread(
                    target=TRexGaming.train_thread,
                    args=[self.sess, train_op, global_step, loss, inputs, self.prepared_queue,
                          self.batch_size, max_steps, save_steps, training_saver, self.dev, self.replay_size])
            
                train_thread.daemon = True
                train_thread.start()

    def take_a_action(self, curr_state):
        q_value, action = self.sess.run(
                                [self.pred_q_value, self.pred_action],
                                feed_dict={self.curr_state_input:curr_state})
            #print(q_value, action)
        return q_value, action

    def restart(self):
        pass