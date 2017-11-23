"""
Reference: `Queue runner impl <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/queue_runner_impl.py>`_
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import weakref
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import queue_runner as qr


class GeneralQueueRunner(qr.QueueRunner):
    def __init__(self, queue=None, enqueue_ops=None, close_op=None,
                 cancel_op=None, queue_closed_exception_types=None,
                 queue_runner_def=None, import_scope=None,
                 feed_dict_funcs=None, feed_tensors=None):
        """Create a GeneralQueueRunner

        :param queue: Queue Object
        :param enqueue_ops: Enqueue operators
        :param close_op: Close operator
        :param cancel_op: Cancel operator
        :param queue_closed_exception_types: Tuple of exception types, which indicate
                the queue has been safely closed.
        :param queue_runner_def: Queue runner from graph def
        :param import_scope: Scope to import
        :param feed_dict_funcs: Feed data function
        :param feed_tensors: Feed tensor
        :return:

        """

        if queue_runner_def:
            if queue or enqueue_ops:
                raise ValueError("queue_runner_def and queue are mutually exclusive.")
            self._init_from_proto(queue_runner_def,
                                  import_scope=import_scope)
        else:
            self._init_from_args(
                queue=queue, enqueue_ops=enqueue_ops,
                close_op=close_op, cancel_op=cancel_op,
                queue_closed_exception_types=queue_closed_exception_types,
                feed_dict_funcs=feed_dict_funcs, feed_tensors=feed_tensors
            )

            # Protect the count of runs to wait for.
            self._lock = threading.Lock()
            # A map from a session object to the number of outstanding queue runner
            # threads for that session.
            self._runs_per_session = weakref.WeakKeyDictionary()
            self._exceptions_raised = []

    def _init_from_args(self, queue=None, enqueue_ops=None, close_op=None,
                        cancel_op=None, queue_closed_exception_types=None,
                        feed_dict_funcs=None, feed_tensors=None):
        """Create a QueueRunner from arguments

        :param queue: A `Queue`
        :param enqueue_ops: List of enqueue ops to run in threads later.
        :param close_op: Op to close the queue. Pending enqueue ops are preserved.
        :param cancel_op:  Op to close the queue and cancel pending enqueue ops.
        :param queue_closed_exception_types: Tuple of exception types, which indicate
                the queue has been safely closed.
        :param feed_dict_fn: Function to get data into queue
        :param feed_tensor: Tensor to feed
        :return:
        """
        if not queue or not enqueue_ops:
            raise ValueError("Must provide queue and enqueue_ops")
        self._queue = queue
        self._enqueue_ops = enqueue_ops
        self._close_op = close_op
        self._cancel_op = cancel_op
        self._feed_dict_funcs = feed_dict_funcs
        self._feed_tensors = feed_tensors

        if queue_closed_exception_types is not None:
            if (not isinstance(queue_closed_exception_types, tuple)
                or not queue_closed_exception_types
                or not all(issubclass(t, errors.OpError)
                           for t in queue_closed_exception_types)):
                raise TypeError(
                        "queue_closed_exception_types, when provided, "
                        "must be a tuple of tf.error types, but saw: %s"
                        % queue_closed_exception_types)

        self._queue_closed_exception_types = queue_closed_exception_types

        # Close when no more will be produced, but pending enqueues should be
        # preserved.
        if self._close_op is None:
            self._close_op = self._queue.close()

        # Close and cancel pending enqueues since there was an error and we want
        # to unblock everything so we can cleanly exit.
        if self._cancel_op is None:
            self._cancel_op = self._queue.close(cancel_pending_enqueues=True)
        if not self._queue_closed_exception_types:
            self._queue_closed_exception_types = (errors.OutOfRangeError,)
        else:
            self._queue_closed_exception_types = tuple(
                self._queue_closed_exception_types
            )

    def _init_from_proto(self, queue_runner_def, import_scope=None):
        """Create a QueueRunner from `QueueRunnerDef`.

        :param queue_runner_def: Optional `QueueRunnerDef` protocol buffer.
        :param import_scope: Optional `string`. Name scope to add.
        """
        raise NotImplementedError(
            "{} does not support initialization from proto.".format(type(
                self).__name__))

    @property
    def queue(self):
        return self._queue

    @property
    def enqueue_ops(self):
        return self._enqueue_ops

    @property
    def close_op(self):
        return self._close_op

    @property
    def cancel_op(self):
        return self._cancel_op

    @property
    def queue_closed_exception_types(self):
        return self._queue_closed_exception_types

    @property
    def exception_raised(self):
        """Exceptions raised but not handled by the `QueueRunner` threads.

        Exceptions raised in queue runner threads are handled in one of two ways
        depending on whether or not a `Coordinator` was passed to
        `create_threads()`:
        * With a `Coordinator`, exceptions are reported to the coordinator and
          forgotten by the `QueueRunner`.
        * Without a `Coordinator`, exceptions are captured by the `QueueRunner` and
          made available in this `exceptions_raised` property.
        :return:
          A list of Python `Exception` objects.  The list is empty if no exception
          was captured.  (No exceptions are captured when using a Coordinator.)
        """
        return self._exceptions_raised

    @property
    def name(self):
        """The string name of the underlying Queue."""
        return self._queue.name


    # pylint: disable=broad-except
    def _run(self, sess, enqueue_op, coord=None, feed_dict_fn=None, feed_tensor=None):
        """Execute the enqueue op in a loop, close the queue in case of error.

        :param sess: A Session
        :param enqueue_op: The Operation to run
        :param coord: Optional Coordinator object for reporting errors and checking
        :param feed_dict_fn: A function to get data that corresponds to a enqueue_op
        :param feed_tensor: A tensor to feed data which return from feed_dict_fn
        :return:
        """

        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    if not feed_dict_fn is None and not feed_tensor is None:
                        sess.run(enqueue_op, feed_dict={feed_tensor: feed_dict_fn()})
                    else:
                        sess.run(enqueue_op)
                except self._queue_closed_exception_types:  # pylint: disable=catching-non-exception
                    # This exception indicates that a queue was closed
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except Exception as e:
            # This catches all other exceptions.
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in GeneralQueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1

    def _close_on_stop(self, sess, cancel_op, coord):
        """Close the queue when the Coordinator requests stop

        :param sess: A session
        :param cancel_op: The Operation to run
        :param coord: Coordinator
        :return:
        """
        coord.wait_for_stop()
        try:
            sess.run(cancel_op)
        except Exception as e:
            # Intentionally ignore errors from cancel_op.
            logging.vlog(1, "Ignored exception: %s", str(e))
    # pylint: enable=broad-except

    def create_threads(self, sess, coord=None, daemon=False, start=False):
        """Create threads to run the enqueue ops for the given session.

        This method requires a session in which the graph was launched.  It creates
        a list of threads, optionally starting them.  There is one thread for each
        op passed in `enqueue_ops`.
        The `coord` argument is an optional coordinator that the threads will use
        to terminate together and report exceptions.  If a coordinator is given,
        this method starts an additional thread to close the queue when the
        coordinator requests a stop.
        If previously created threads for the given session are still running, no
        new threads will be created.

        :param sess: A `Session`
        :param coord: Optional `Coordinator` object for reporting errors and checking
            stop conditions.
        :param daemon: Boolean. If `True` make the threads daemon threads.
        :param start: Boolean If `True` starts the threads. If `False` the
            caller must call the `start()` method of the returned threads
        :return:
            A list of threads
        """
        with self._lock:
            try:
                if self._runs_per_session[sess] > 0:
                    # Already started: no new threads to return
                    return []
            except KeyError:
                # We haven't seen this session yet
                pass
            self._runs_per_session[sess] = len(self._enqueue_ops)
            self._exceptions_raised = []

        if self._feed_dict_funcs is None or self._feed_tensors is None:
            ret_threads = [threading.Thread(target=self._run, args=(sess, op, coord))
                           for op in self._enqueue_ops]
        else:
            ret_threads = [threading.Thread(target=self._run,
                                            args=(sess, op, coord, feed_dict_fn, feed_tensor))
                           for op, feed_dict_fn, feed_tensor in
                           zip(self._enqueue_ops, self._feed_dict_funcs, self._feed_tensors)]
        if coord:
            ret_threads.append(threading.Thread(target=self._close_on_stop,
                                                args=(sess, self._cancel_op, coord)))

        for t in ret_threads:
            if coord:
                coord.register_thread(t)
            if daemon:
                t.daemon = True
            if start:
                t.start()

        return ret_threads

    def to_proto(self, export_scope=None):
        """Converts this `GeneralQueueRunner` to a `GeneralQueueRunnerDef` protocol buffer.

        :param export_scope: Optional `string`. Name scope to remove.

        :return:
            A `QueueRunnerDef` protocol buffer, or `None` if the `Variable` is not in
            the specified name scope.
        """
        raise NotImplementedError(
            "{} does not support serialization to proto.".format(type(
                self).__name__))

    @staticmethod
    def from_proto(queue_runner_def, import_scope=None):
        """Returns a `GeneralQueueRunner` object created from `queue_runner_def`."""
        return GeneralQueueRunner(queue_runner_def=queue_runner_def,
                           import_scope=import_scope)

# ops.register_proto_function('GENERAL_QUEUE_RUNNERS',
#                             proto_type=queue_runner_pb2.QueueRunnerDef,
#                             to_proto=GeneralQueueRunner.to_proto,
#                             from_proto=GeneralQueueRunner.from_proto)