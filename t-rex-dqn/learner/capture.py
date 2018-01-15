#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
from wsgiref.simple_server import make_server
from socketio.server import SocketIOServer
from pyramid.paster import get_app
from gevent import monkey; monkey.patch_all()
from pyramid.config import Configurator
from pyramid.response import Response
import cv2
import numpy as np
from socketio import socketio_manage
from socketio.namespace import BaseNamespace
from socketio.mixins import BroadcastMixin
from t_rex_tf import TRexGaming
import threading
import scipy.misc
import random
from collections import deque


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("checkpoint", None, "Path of checkpoint file")
tf.app.flags.DEFINE_string("mode","learn", "Learn or play")
tf.app.flags.DEFINE_float("epsilon", 0.1, "Epsilon greedy initial value")
tf.app.flags.DEFINE_float("lrn_rate", 1e-6, "Initial learning rate")
tf.app.flags.DEFINE_boolean("using_cuda", True, "Use single GPU for running")
tf.app.flags.DEFINE_float("discount", 0.95, "Discount factor")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size for a learning step")
tf.app.flags.DEFINE_integer("replay_size", 50000, "Replay memory size")
tf.app.flags.DEFINE_integer("save_steps", 50000, "Save steps")
tf.app.flags.DEFINE_integer("debug", False, "Debug flag default is false")

KEYCODES = {
    'JUMP': ['38', '32'],
    'DUCK': ['40'],
    'RESTART': ['13']
}

FRAME_QUEUE = []
TRAIN_DATA_QUEUE = deque()
GAME_OVER = 0
GAME_PLAYING = 1
FRAME_CHANNELS = 4
T_REX = None
GAME_TURN = 0
NUM_FRAME = 0


class BotNamespace(BaseNamespace, BroadcastMixin):
    def __init__(self, *args, **kwargs):
        super(BotNamespace, self).__init__(*args, **kwargs)
        self.frame_queue = []
        self.raw_frame_queue = []
        self.last_frame_tuple = None
        self.prev_state = None
        self.curr_state = None
        self.debug = True
        self.emit_action = True
        self.dispatch_action = 'NONE'

    def decode_action(self, action):
        if action == 'NONE': 
            return 0
        if action == 'JUMP': 
            return 1
        # else: #action == 'DUCK'
        #     return 2
    def num_to_action(self, action):
        if action == 0:
            return 'NONE'
        if action == 1:
            return 'JUMP'
        # if action == 2:
        #   return 'DUCK'

    def encode_action(self, action):
        if action == 0:
            return 0
        if action == 1:
            return '38' # Javascript keyCode for the up arrow key
        # else:
        #     return '40' # Javascript keyCode for the down arrow key

    def combine_frames(self, frames):
        c0 = frames[0]
        c1 = frames[1]
        c2 = frames[2]
        c3 = frames[3]
        s = np.stack([c0, c1, c2, c3], axis=2)
        return s

    def on_recv_frame(self, data):
        global GAME_TURN, NUM_FRAME, T_REX
        frame = data['frame']
        #action = data['action']
        game_status = data['game_status']
        t_rex_status = data['t_rex_status']
        # Decode frame and put it in queue
        frame = frame.replace('data:image/png;base64,', '')
        frame += '='
        frame = frame.decode("base64")
        frame = np.fromstring(frame, np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, gray_frame = cv2.threshold(gray_frame,127,255,cv2.THRESH_BINARY)
        resized_frame = cv2.resize(gray_frame, (200, 50))

        print_game_turn = False
        terminal = 0
        reward = 1.
        if game_status == GAME_OVER:
            print_game_turn = True
            GAME_TURN += 1
            reward = -100.
            terminal = 1

        # Skipping jumping frames
        if t_rex_status == 1 and game_status != GAME_OVER:
            return None

        NUM_FRAME += 1

        self.frame_queue.append(resized_frame)
        self.raw_frame_queue.append(gray_frame)

        if len(self.frame_queue) == 4:
            self.prev_state = self.curr_state
            self.curr_state = self.combine_frames(self.frame_queue)

            if FLAGS.mode == "learn":
                if not self.prev_state is None:
                    if len(TRAIN_DATA_QUEUE) > FLAGS.replay_size:
                        if TRAIN_DATA_QUEUE[0][-1] != 1:
                            TRAIN_DATA_QUEUE.popleft()
                        else:
                            first = TRAIN_DATA_QUEUE.popleft()
                            ep = random.random()
                            if ep < 0.8:
                                TRAIN_DATA_QUEUE.append(first)

                    if FLAGS.debug:
                        scipy.misc.imsave('%d_prev.jpg' % NUM_FRAME, self.raw_frame_queue[-2])
                        scipy.misc.imsave('%d_curr.jpg' % NUM_FRAME, self.raw_frame_queue[-1])
                        print("FRAME:%d - REWARD: %0.4f" % (NUM_FRAME, reward))
                    ep = random.random()
                    is_append = True
                    if ep < 0.3 and game_status != GAME_OVER:
                        is_append = False

                    if is_append:
                        TRAIN_DATA_QUEUE.append([
                                self.prev_state, 
                                self.decode_action(self.dispatch_action), 
                                reward, 
                                self.curr_state, 
                                terminal])

            self.raw_frame_queue.pop(0)
            self.frame_queue.pop(0)

        if not self.curr_state is None:
            if game_status == GAME_PLAYING:
                ext_curr_state = self.curr_state[np.newaxis,:,:,:]
                qvalue, next_action = T_REX.take_a_action(ext_curr_state)
                next_action = int(next_action)
                self.dispatch_action = self.num_to_action(next_action)
                #print('qvalue:', qvalue.tolist(), 'action:', next_action)
                next_action = self.encode_action(next_action)
            else:
                next_action = "13"
                self.dispatch_action = 'JUMP'
                self.prev_state = None
                self.curr_state = None
                self.frame_queue = []
                self.raw_frame_queue = []

            if GAME_TURN > 0 and GAME_TURN % 5 == 0 and print_game_turn:
                print("Game turn:%d" % GAME_TURN)

            if self.emit_action:
                self.emit('recv_action', {"action": next_action})


    def recv_connect(self):
        self.dispatch_action = 'JUMP'
        self.emit('recv_action', {"action": "38"})


def socketio_service(request):
    socketio_manage(request.environ,
                    {'/bot': BotNamespace},
                    request=request)

    return Response('')

if __name__ == '__main__':
    try:
        T_REX = TRexGaming(
            TRAIN_DATA_QUEUE, 
            batch_size=FLAGS.batch_size,
            mode=FLAGS.mode,
            discount=FLAGS.discount,
            epsilon=FLAGS.epsilon,
            using_cuda=FLAGS.using_cuda,
            replay_size=FLAGS.replay_size)
        T_REX.running(checkpoint=FLAGS.checkpoint, save_steps=FLAGS.save_steps)
        config = Configurator()
        config.add_route('socket_io', 'socket.io/*remaining')
        config.add_view(socketio_service, route_name='socket_io')
        config.add_static_view('/', 't_rex')
        app = config.make_wsgi_app()

        SocketIOServer(('0.0.0.0', 1234), app,
                       resource="socket.io", policy_server=True,
                       policy_listener=('0.0.0.0', 10843)).serve_forever()

    except KeyboardInterrupt:
        sys.exit(0)
    
