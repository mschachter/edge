import numpy as np

import tensorflow as tf
from tensorflow.python.ops import math_ops

import layers


NET_TYPES = {'SRNN':layers.SRNN_Layer,
             'LSTM':layers.LSTM_Layer,
             'GRU':layers.GRU_Layer,
             'EDSRNN':layers.EDSRNN_Layer,
             'EDGRU':layers.EDGRU_Layer,
             'EI':layers.EI_Layer}


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

    def l2_R_elementwise(self, lambda_mat, lambda_b=1.):
        # print("lambda_mat.shape=" + str(lambda_mat.shape))
        # print("self.rnn_layer.R.get_shape()=" + str(self.rnn_layer.R.get_shape()))
        assert lambda_mat.shape == self.rnn_layer.R.get_shape()
        assert np.isscalar(lambda_b)
        # return tf.reduce_mean(tf.mul(lambda_mat, tf.square(self.rnn_layer.R))) + lambda_b*tf.reduce_mean(tf.square(self.rnn_layer.b))
        return lambda_b*(tf.reduce_mean(tf.square(self.rnn_layer.R)) + tf.reduce_mean(tf.square(self.rnn_layer.b)))

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
            M = math_ops.maximum(tf.mul(S, G), 0, name='M')

            # constrain weights to be positive
            return self.rnn_layer.R.assign(tf.mul(self.rnn_layer.R, M))

    def sign_cost(self, sign_mat, sign_lambda=1.):

        # encode the sign matrix as a graph variable
        S = tf.constant(sign_mat.astype('float32'))

        # get the sign of the recurrent net weights
        R = self.rnn_layer.R

        # compute the cost for each weight
        C = 0.5 * S * R * (S*R - tf.abs(R))

        # the total cost is the mean
        return sign_lambda*tf.reduce_mean(C)

    def distance_constrain_R(self, dist_mat, cutoff=500e-3, min_val=0., name=None):
        with tf.op_scope([dist_mat], name, "distance_constrain_R") as scope:
            # zero out weights that are further than the cutoff
            M = (dist_mat <= cutoff).astype('float32')
            Z = tf.mul(self.rnn_layer.R, M)
            if min_val > 0:
                Rnew = Z + (1-M)*min_val
            else:
                Rnew = Z

            return self.rnn_layer.R.assign(Rnew)

    def activity_cost(self, h, a, deg=2):
        if deg == 2:
            return tf.reduce_mean(tf.matmul(tf.square(h), a))
        elif deg == 1:
            return tf.reduce_mean(tf.matmul(tf.abs(h), a))

    def rescale_R(self, scale, name=None):

        with tf.op_scope([scale], name, "rescale_R") as scope:
            return self.rnn_layer.R.assign(scale*self.rnn_layer.R)


class Deep_Recurrent_Network():

    def __init__(self, hparams):

        self.hparams = hparams
        self.layers = list()

        n_total_units = 0
        # construct each layer according to the configuration options specified in hparams
        n_in = self.hparams['n_in']
        for k,ldict in enumerate(self.hparams['layers']):
            with tf.name_scope('layer%d' % k) as scope:
                # make the layer, which should be responsible for reading the relevant options from the configuration
                assert ldict['rnn_type'] in NET_TYPES
                rnn_class = NET_TYPES[ldict['rnn_type']]
                rnn_layer = rnn_class(n_in, ldict['n_unit'], ldict)
                self.layers.append(rnn_layer)
                n_total_units += ldict['n_unit']

            # specify the number of inputs for the next layer
            n_in = ldict['n_unit']

        # construct the output layer
        with tf.name_scope('output_layer') as scope:
            self.output_layer = layers.Linear_Layer(n_total_units, hparams['n_out'], hparams)

    def step(self, state, x_input):
        """
        :param state: A list of state vectors, one state vector for each layer. Each state vector is of
                      size (num_batches, num_units_in_layer).
        :param x_input: The input to the network.
        """

        with tf.name_scope('step') as scope:
            # run the input through each layer
            the_input = x_input
            layer_states = list()
            for k,layer in enumerate(self.layers):
                (layer_state,) = layer.step((state[k],), the_input)
                layer_states.append(layer_state)
                the_input = layer_state

            # concatenate the hidden states together into one vector
            z = tf.concat(1, layer_states)

            # run the output layer
            output = self.output_layer.output(z)

        return layer_states, output

    def activity_cost(self, state):
        return 0.

    def weight_cost(self):
        pass
