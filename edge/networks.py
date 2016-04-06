import numpy as np

import tensorflow as tf
from tensorflow.python.ops import math_ops

import layers


NET_TYPES = {'SRNN':layers.SRNN_Layer,
             'LSTM':layers.LSTM_Layer,
             'GRU':layers.GRU_Layer,
             'EDSRNN':layers.EDSRNN_Layer,
             'EDGRU':layers.EDGRU_Layer}


class Basic_Network(object):

    def __init__(self, n_input, hparams, n_output=None):

        n_unit = hparams['n_unit']

        if n_output is None:
            n_output = 1

        # construct the recurrent layer
        assert hparams['rnn_type'] in NET_TYPES
        rnn_class = NET_TYPES[hparams['rnn_type']]
        self.rnn_layer = rnn_class(n_input, n_unit, hparams)

        # construct the output layer
        with tf.name_scope('output_layer') as scope:
            self.output_layer = layers.Linear_Layer(n_unit, n_output, hparams)

        # determine whether recurrent layer gets error information as input
        self.uses_error = False
        if hparams['rnn_type'].startswith('ED'):
            self.uses_error = True

    def step(self, state, x_input, *d_state, **kwargs):
        state = self.rnn_layer.step(state, x_input, *d_state, **kwargs)
        output = self.output_layer.output(state[0])
        return state, output

    def gradient(self, error, state):
        """ Computes the gradient of the error with respect to the network state """
        return self.rnn_layer.gradient(error, state)

    def get_new_states(self, n_state):
        return self.rnn_layer.get_new_states(n_state)

    def store_state_op(self, state, storage):
        """ Records current state of the network in storage """
        store_ops = [store_var.assign(state_var) for state_var, store_var in zip(state, storage)]
        return tf.group(*store_ops)

    def reset_state_op(self, state):
        """ Resets the network state to zero """
        reset_ops  = [state_var.assign(tf.zeros(state_var.get_shape())) for state_var in state]
        return tf.group(*reset_ops)

    def l2_W(self, lambda_val=1.):
        return lambda_val*tf.reduce_mean(tf.square(self.rnn_layer.W))

    def l2_R(self, lambda_val=1.):
        return lambda_val*(tf.reduce_mean(tf.square(self.rnn_layer.R)) + tf.reduce_mean(tf.square(self.rnn_layer.b)))

    def l2_Wout(self, lambda_val=1.):
        return lambda_val*(tf.reduce_mean(tf.square(self.output_layer.W)) + tf.reduce_mean(tf.square(self.output_layer.b)))

    def sign_constraint_R(self, sign_mat, name=None):
        """ Constrain the recurrent weight matrix so that each element has the specified sign. Returns an op that zeros
            out weights that are the wrong sign.

        :param sign_mat: A matrix of shape (n_hidden, n_hidden) that has a 1 for elements that should be positive,
                and -1 for elements that should be negative.
        """

        with tf.op_scope([sign_mat], name, "sign_constrain_R") as scope:
            # encode the sign matrix as a graph variable
            G = tf.constant(sign_mat.astype('float32'))

            # get the sign of the recurrent net weights
            S = tf.sign(self.rnn_layer.R)

            # compute a binary mask that will zero out weights that are the wrong sign
            M = math_ops.maximum(tf.mul(S, G), 0)

            # constrain weights to be positive
            return self.rnn_layer.R.assign(tf.mul(self.rnn_layer.R, M))
