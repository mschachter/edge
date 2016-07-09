import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


def init_weights(n_input, n_unit, hparams, scale=1.):

    # Every ones knows this is how you do it of course
    std = scale*(tf.sqrt(2.0) / tf.sqrt(tf.cast(n_input + n_unit, tf.float32)))
    return tf.truncated_normal([n_input, n_unit], 0.0, std)


def init_complex_weights(n_input, n_unit, hparams):

    std = tf.sqrt(2.0) / tf.sqrt(tf.cast(n_input + n_unit, tf.float32))
    X = tf.truncated_normal([n_input, n_unit], 0.0, std)
    Y = tf.truncated_normal([n_input, n_unit], 0.0, std)

    return tf.complex(X, Y)


class Linear_Layer(object):
    def __init__(self, n_input, n_output, hparams):
        self.n_input = n_input
        self.n_unit = n_output

        self.W = tf.Variable(init_weights(n_input, n_output, hparams), name= 'W')
        self.b = tf.Variable(tf.zeros([n_output]), name = 'b')

        self.hparams = hparams

    def output(self, x):
        v = tf.nn.xw_plus_b(x, self.W, self.b)
        return v

    def get_saveable_params(self, session):
        cnames = ['W', 'b', ]
        to_compute = [self.W, self.b]
        cvals = session.run(to_compute)
        return {k: v for k, v in zip(cnames, cvals)}

    def weight_cost(self):
        if 'output_lambda2' in self.hparams and self.hparams['output_lambda2'] > 0:
            l2_W = tf.reduce_mean(tf.square(self.W))
            l2_b = tf.reduce_mean(tf.square(self.b))
            return self.hparams['output_lambda2']*(l2_W + l2_b)
        else:
            return 0.0


class ComplexOutput_Layer(object):
    def __init__(self, n_input, n_output, hparams):
        self.n_input = n_input
        self.n_unit = n_output

        self.W = tf.Variable(init_complex_weights(n_input, n_output, hparams), name= 'W')
        self.b = tf.Variable(tf.complex(tf.zeros([n_output]), tf.zeros([n_output])), name = 'b')

        self.hparams = hparams

    def output(self, x):
        v = tf.nn.xw_plus_b(x, self.W, self.b)
        return tf.real(v)

    def get_saveable_params(self, session):
        cnames = ['W', 'b', ]
        to_compute = [self.W, self.b]
        cvals = session.run(to_compute)
        return {k: v for k, v in zip(cnames, cvals)}

    def weight_cost(self):
        return 0.0


class FeedforwardLayer(object):

    def __init__(self, n_input, n_output, hparams):

        self.hparams = hparams

        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = hparams['output_n_hidden']

        assert hparams['output_activation'] in ['sigmoid', 'tanh', 'relu', 'elu', 'linear']
        if hparams['output_activation'] == 'linear':
            self.activation = lambda x: x
        else:
            self.activation = getattr(tf.nn, hparams['output_activation'])

        self.W_in = tf.Variable(init_weights(self.n_input, self.n_hidden, hparams), name='Win')
        self.b_in = tf.Variable(tf.zeros([self.n_hidden]), name='bin')

        self.W = tf.Variable(init_weights(self.n_hidden, self.n_output, hparams), name='W')
        self.b = tf.Variable(tf.zeros([self.n_output]), name='b')

    def output(self, x):
        u = self.activation(tf.matmul(x, self.W_in) + self.b_in)
        return tf.matmul(u, self.W) + self.b

    def get_saveable_params(self, session):
        cnames = ['W', 'W_in', 'b', 'b_in']
        to_compute = [self.W, self.W_in, self.b, self.b_in]
        cvals = session.run(to_compute)

        return {k:v for k,v in zip(cnames, cvals)}

    def weight_cost(self):
        if 'output_lambda2' in self.hparams and self.hparams['output_lambda2'] > 0:
            l2_W = tf.reduce_mean(tf.square(self.W))
            l2_b = tf.reduce_mean(tf.square(self.b))
            l2_W_in = tf.reduce_mean(tf.square(self.W_in))
            l2_b_in = tf.reduce_mean(tf.square(self.b_in))
            return self.hparams['output_lambda2'] * (l2_W + l2_b + l2_W_in + l2_b_in)
        return 0.


class SRNN_Layer(object):

    def __init__(self, n_input, n_unit, hparams):
        self.n_input = n_input
        self.n_unit = n_unit
        self.hparams = hparams

        with tf.name_scope('srnn_layer'):
            self.W = tf.Variable(init_weights(n_input, n_unit, hparams), name = 'W')
            self.R = tf.Variable(init_weights(n_unit, n_unit, hparams), name = 'R')
            self.b = tf.Variable(tf.zeros([1, n_unit]), name = 'b')

        if 'activation' in hparams:
            assert hparams['activation'] in ['sigmoid', 'tanh', 'relu', 'elu', 'linear']
            if hparams['activation'] == 'linear':
                self.activation = lambda x: x
            else:
                self.activation = getattr(tf.nn, hparams['activation'])
        else:
            self.activation = tf.nn.tanh

    def get_new_states(self, n_state):
        new_h = tf.Variable(tf.zeros([n_state, self.n_unit]), trainable=False, name = 'h')
        return new_h,

    def initial_state(self, n_batches):
        return np.random.randn(n_batches, self.n_unit)

    def step(self, state, x, *d_state, **kwargs):
        """Updates returns the state updated by input x"""
        h = state[0]
        W = self.W
        R = self.R
        xxx = tf.matmul(x, W)
        hhh = tf.matmul(h, R)
        h = self.activation(xxx + hhh + self.b)

        return h,

    def activity_cost(self, state):
        if 'activity_lambda' in self.hparams:
            activity_deg = 2.
            if 'activity_deg' in self.hparams:
                activity_deg = self.hparams['activity_deg']
            nonlin = tf.square
            if activity_deg == 1:
                nonlin = tf.abs
            return tf.reduce_mean(nonlin(state))*self.hparams['activity_lambda']

        return 0.

    def get_saveable_params(self, session):
        to_compute = [self.W, self.R, self.b]
        vals = session.run(to_compute)
        params = dict()
        for k,t in enumerate(to_compute):
            tname = t.name.split(':')[0]
            tname = tname.split('/')[-1]
            params[tname] = vals[k]

        for k,v in self.hparams.items():
            if np.isscalar(v) or isinstance(v, str):
                params[k] = v

        return params

    def weight_cost(self):
        total_cost = tf.constant(0.)
        if 'lambda2' in self.hparams and self.hparams['lambda2'] > 0:
            l2_W = tf.reduce_mean(tf.square(self.W))
            l2_R = tf.reduce_mean(tf.square(self.R))
            l2_b = tf.reduce_mean(tf.square(self.b))
            total_cost += self.hparams['lambda2']*(l2_W + l2_R + l2_b)
        if 'lambda1' in self.hparams and self.hparams['lambda1'] > 0:
            l1_R = tf.reduce_mean(tf.abs(self.R))
            total_cost += self.hparams['lambda1']*l1_R
        return total_cost

    def gradient(self, error, state):
        return tf.gradients(error, state[0])

    @classmethod
    def plot(clz, params=None):

        # get current values of weights and bias terms
        Wnow = params['W']
        Rnow = params['R']
        bnow = params['b']

        figsize = (5, 13)
        plt.figure(figsize=figsize)
        gs = plt.GridSpec(100, 1)

        ax = plt.subplot(gs[:35, 0])
        absmax = np.abs(Wnow).max()
        plt.imshow(Wnow, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
        plt.title('W')

        ax = plt.subplot(gs[45:80, 0])
        absmax = np.abs(Rnow).max()
        plt.imshow(Rnow, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
        plt.title('R')

        ax = plt.subplot(gs[85:, 0])
        absmax = np.abs(bnow).max()
        plt.axhline(0, c='k')
        n_unit = len(bnow.squeeze())
        plt.bar(range(n_unit), bnow.squeeze(), color='k', alpha=0.7)
        plt.axis('tight')
        plt.ylim(-absmax, absmax)
        plt.title('b')


class EI_Layer(object):

    def __init__(self, n_input, n_unit, hparams, input_weight_mask=None, input_sign=None):
        self.n_input = n_input
        self.n_unit = n_unit
        self.hparams = hparams

        assert 'sign' in hparams, "Must supply 'sign' in hparams, a vector of length n_unit that has 1 for excitatory neurons, -1 for inhibitory neurons"
        self.sign = hparams['sign']
        assert len(self.sign) == self.n_unit, "Wrong size for hparams['sign'], must be of length %d" % n_unit

        M = np.ones([n_unit, n_unit])
        if 'mask' in self.hparams:
            M = self.hparams['mask']
        assert M.shape == (n_unit, n_unit)

        with tf.name_scope('ei_layer'):

            self.Mr = tf.constant(M.astype('float32'), name='Mr')
            self.Dr = tf.constant(np.diag(self.sign.astype('float32')), name='Dr')
            self.Jr = tf.Variable(init_weights(n_unit, n_unit, self.hparams), name='Jr', trainable=True)

            self.b = tf.Variable(tf.zeros([1, n_unit]), name='b', trainable=True)

            self.R = tf.matmul(self.Dr, tf.nn.relu(self.Jr)) * self.Mr
            # self.R.name = 'R'

            # optionally mask the input weight matrix
            Mw = np.ones([n_input, n_unit])
            if input_weight_mask is not None:
                assert Mw.shape == (n_input, n_unit)
                Mw = input_weight_mask
            self.Mw = tf.constant(Mw.astype('float32'), name='Mw')

            # initialize the input weight matrix, with potential sign constraints
            if input_sign is None:
                self.Jw = tf.Variable(init_weights(n_input, n_unit, self.hparams), name='Jw', trainable=True)
                self.W = self.Mw * self.Jw
            else:
                assert len(input_sign) == n_unit
                self.Jw = tf.Variable(init_weights(n_input, n_unit, self.hparams), name='Jw', trainable=True)
                self.Dw = tf.constant(np.diag(input_sign).astype('float32'), name='Dw')
                self.W = tf.matmul(self.Dw, tf.nn.relu(self.Jw)) * self.Mw

        if 'activation' in self.hparams:
            assert self.hparams['activation'] in ['sigmoid', 'relu', 'elu']
            self.activation = getattr(tf.nn, hparams['activation'])
        else:
            self.activation = tf.nn.relu

    def get_new_states(self, n_state):
        new_h = tf.Variable(tf.zeros([n_state, self.n_unit]), trainable=False, name='h')
        return new_h,

    def initial_state(self, n_batches):
        h = np.random.randn(n_batches, self.n_unit)
        h[h < 0] = 0
        h *= 1e-1
        return h

    def step(self, state, x, *d_state, **kwargs):
        """Updates returns the state updated by input x"""
        h = state[0]

        xxx = tf.matmul(x, self.W)
        hhh = tf.matmul(h, self.R)
        h = self.activation(xxx + hhh + self.b)

        return h,

    def gradient(self, error, state):
        return tf.gradients(error, state[0])

    def activity_cost(self, state):
        if 'activity_lambda' in self.hparams:
            activity_deg = 2.
            if 'activity_deg' in self.hparams:
                activity_deg = self.hparams['activity_deg']
            nonlin = tf.square
            if activity_deg == 1:
                nonlin = tf.abs
            return tf.reduce_mean(nonlin(state)) * self.hparams['activity_lambda']

        return 0.

    def weight_cost(self):
        total_cost = tf.constant(0.)
        if 'lambda2' in self.hparams and self.hparams['lambda2'] > 0:
            l2_W = tf.reduce_mean(tf.square(self.Jw))
            l2_R = tf.reduce_mean(tf.square(self.Jr))
            l2_b = tf.reduce_mean(tf.square(self.b))
            total_cost += self.hparams['lambda2'] * (l2_W + l2_R + l2_b)
        if 'lambda1' in self.hparams and self.hparams['lambda1'] > 0:
            l1_R = tf.reduce_mean(tf.abs(self.Jw))
            total_cost += self.hparams['lambda1'] * l1_R
        return total_cost

    def get_saveable_params(self, session):
        to_compute = [self.W, self.R, self.b, self.Jw, self.Jr]
        val_names = ['W', 'R', 'b', 'Jw', 'Jr']
        vals = session.run(to_compute)

        params = dict()
        for k, t in enumerate(to_compute):
            params[val_names[k]] = vals[k]

        for k, v in self.hparams.items():
            if np.isscalar(v) or isinstance(v, str):
                params[k] = v

        return params

    def plot(clz, params=None):

        # get current values of weights and bias terms
        Jw = params['Jw']
        W = params['W']
        Jr = params['Jr']
        R = params['R']
        b = params['b']

        figsize = (15, 13)
        plt.figure(figsize=figsize)
        gs = plt.GridSpec(100, 2)

        ax = plt.subplot(gs[:35, 0])
        absmax = np.abs(Jw).max()
        plt.imshow(Jw, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
        plt.title('Jw')
        plt.colorbar()

        ax = plt.subplot(gs[:35, 1])
        absmax = np.abs(W).max()
        plt.imshow(W, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
        plt.title('W')
        plt.colorbar()

        ax = plt.subplot(gs[45:80, 0])
        absmax = np.abs(Jr).max()
        plt.imshow(Jr, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
        plt.title('Jr')
        plt.colorbar()

        ax = plt.subplot(gs[45:80, 1])
        absmax = np.abs(R).max()
        plt.imshow(R, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
        plt.title('R')
        plt.colorbar()

        ax = plt.subplot(gs[85:, 0])
        absmax = np.abs(b).max()
        plt.axhline(0, c='k')
        n_unit = len(b.squeeze())
        plt.bar(range(n_unit), b.squeeze(), color='k', alpha=0.7)
        plt.axis('tight')
        plt.ylim(-absmax, absmax)
        plt.title('b')


class ComplexLinear_layer(object):

    def __init__(self, n_input, n_unit, hparams):
        self.n_input = n_input
        self.n_unit = n_unit
        self.hparams = hparams

        with tf.name_scope('complex_layer'):
            self.W = tf.Variable(init_complex_weights(n_input, n_unit, hparams), name='W')
            self.R = tf.Variable(init_complex_weights(n_unit, n_unit, hparams), name='R')
            self.b = tf.Variable(tf.complex(np.zeros([1, n_unit]), np.zeros([1, n_unit])), name='b')

    def get_new_states(self, n_state):
        new_h = tf.Variable(tf.zeros([n_state, self.n_unit]), trainable=False, name='h')
        return new_h,

    def initial_state(self, n_batches):
        return np.random.randn(n_batches, self.n_unit)

    def step(self, state, x, *d_state, **kwargs):
        """Updates returns the state updated by input x"""
        h = state[0]
        W = self.W
        R = self.R
        xxx = tf.matmul(x, W)
        hhh = tf.matmul(h, R)

        return hhh + xxx,

    def activity_cost(self, state):
        return 0.

    def get_saveable_params(self, session):
        to_compute = [self.W, self.R, self.b]
        vals = session.run(to_compute)
        params = dict()
        for k, t in enumerate(to_compute):
            tname = t.name.split(':')[0]
            tname = tname.split('/')[-1]
            params[tname] = vals[k]

        for k, v in self.hparams.items():
            if np.isscalar(v) or isinstance(v, str):
                params[k] = v

        return params

    def weight_cost(self):
        return 0.

    def gradient(self, error, state):
        return tf.gradients(error, state[0])

    @classmethod
    def plot(clz, params=None):

        # get current values of weights and bias terms
        Wnow = params['W']
        Rnow = params['R']
        bnow = params['b']

        figsize = (5, 13)
        plt.figure(figsize=figsize)
        gs = plt.GridSpec(100, 1)

        ax = plt.subplot(gs[:35, 0])
        ax.set_axis_background('black')
        plt.imshow(np.abs(Wnow), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot)
        plt.title('W')

        ax = plt.subplot(gs[45:80, 0])
        ax.set_axis_background('black')
        plt.imshow(np.abs(Rnow), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot)
        plt.title('R')

        ax = plt.subplot(gs[85:, 0])
        plt.axhline(0, c='k')
        n_unit = len(bnow.squeeze())
        plt.bar(range(n_unit), np.abs(bnow).squeeze(), color='k', alpha=0.7)
        plt.axis('tight')
        plt.title('b')



class EDSRNN_Layer(SRNN_Layer):

    def __init__(self, n_input, n_unit, hparams):
        super(EDSRNN_Layer, self).__init__(n_input, n_unit, hparams)

        self.E = tf.Variable(init_weights(n_unit, n_unit, hparams), name = 'E')

    def step(self, state, x, *d_state):
        """Updates returns the state updated by input x"""
        dh = d_state[0][0]
        h = state[0]
        # TODO: should there be a non-linearity on the gradient here?????
        state = tf.sigmoid(tf.matmul(x, self.W) + tf.matmul(h, self.R)
            + tf.matmul(dh, self.E) + self.b)

        return state,

class GRU_Layer(object):
    def __init__(self, n_input, n_unit, hparams):
        self.n_input = n_input
        self.n_unit = n_unit

        # The computed update
        with tf.name_scope('update') as scope:
            self.Wu = tf.Variable(init_weights(n_input, n_unit, hparams), name = 'Wu')
            self.Ru = tf.Variable(init_weights(n_unit, n_unit, hparams), name = 'Ru')
            self.bu = tf.Variable(tf.zeros([1, n_unit]), name = 'bu')

        # The reset gate
        with tf.name_scope('reset_gate') as scope:
            self.Wr = tf.Variable(init_weights(n_input, n_unit, hparams), name = 'Wr')
            self.Rr = tf.Variable(init_weights(n_unit, n_unit, hparams), name = 'Rr')
            self.br = tf.Variable(tf.zeros([1, n_unit]), name = 'br')

        # The update gate
        with tf.name_scope('update_gate') as scope:
            self.Wz = tf.Variable(init_weights(n_input, n_unit, hparams), name = 'Wz')
            self.Rz = tf.Variable(init_weights(n_unit, n_unit, hparams), name = 'Rz')
            self.bz = tf.Variable(tf.zeros([1, n_unit]), name = 'bz')

    def get_new_states(self, n_state):
        new_h = tf.Variable(tf.zeros([n_state, self.n_unit]), trainable=False, name = 'h')
        return new_h,

    def step(self, state, x, *d_state):
        h = state[0]

        r = tf.sigmoid(tf.matmul(x, self.Wr) + tf.matmul(h, self.Rr) + self.br)
        z = tf.sigmoid(tf.matmul(x, self.Wz) + tf.matmul(h, self.Rz) + self.bz)

        h_tilde = tf.tanh(tf.matmul(x, self.Wu) + r*tf.matmul(h, self.Ru) + self.bu)

        h = (1.0 - z)*h + z*h_tilde

        return h,

    def gradient(self, error, state):
        return tf.gradients(error, state[0])

class EDGRU_Layer(GRU_Layer):
    def __init__(self, n_input, n_unit, hparams):
        super(EDGRU_Layer, self).__init__(n_input, n_unit, hparams)

        with tf.name_scope('update') as scope:
            self.Eu = tf.Variable(init_weights(n_unit, n_unit, hparams), name = 'Eu')
        with tf.name_scope('reset_gate') as scope:
            self.Er = tf.Variable(init_weights(n_unit, n_unit, hparams), name = 'Er')
        with tf.name_scope('update_gate') as scope:
            self.Ez = tf.Variable(init_weights(n_unit, n_unit, hparams), name = 'Ez')

    def step(self, state, x, *d_state):
        h = state[0]
        dh = d_state[0][0]

        # import ipdb; ipdb.set_trace()

        r = tf.sigmoid(tf.matmul(x, self.Wr) + tf.matmul(h, self.Rr) + tf.matmul(dh, self.Er) + self.br)
        z = tf.sigmoid(tf.matmul(x, self.Wz) + tf.matmul(h, self.Rz) + tf.matmul(dh, self.Ez) + self.bz)

        h_tilde = tf.tanh(tf.matmul(x, self.Wu) + r*(tf.matmul(h, self.Ru) + tf.matmul(dh, self.Eu)) + self.bu)

        h = (1.0 - z)*h + z*h_tilde

        return h,

class LSTM_Layer(object):


    def __init__(self, n_input, n_unit, hparams):
        self.n_input = n_input
        self.n_unit = n_unit

        ## Layer parameters: W input weights, R recurrent weights, b bias

        # The computed update
        with tf.name_scope('update') as scope:
            self.Wu = tf.Variable(init_weights(n_input, n_unit, hparams), name = 'Wu')
            self.Ru = tf.Variable(init_weights(n_unit, n_unit, hparams), name = 'Ru')
            self.bu = tf.Variable(tf.zeros([1, n_unit]), name = 'bu')

        # Input gate
        with tf.name_scope('input_gate') as scope:
            self.Wi = tf.Variable(init_weights(n_input, n_unit, hparams), name = 'Wi')
            self.Ri = tf.Variable(init_weights(n_unit, n_unit, hparams), name = 'Ri')
            self.bi = tf.Variable(tf.zeros([1, n_unit]), name = 'bi')

        # Forget gate
        with tf.name_scope('forget_gate') as scope:
            self.Wf = tf.Variable(init_weights(n_input, n_unit, hparams), name = 'Wf')
            self.Rf = tf.Variable(init_weights(n_unit, n_unit, hparams), name = 'Rf')
            # using a positive bias as suggested in Joxefowicx 2015
            self.bf = tf.Variable(tf.ones([1, n_unit]), name = 'bf')

        # Output gate
        with tf.name_scope('output_gate') as scope:
            self.Wo = tf.Variable(init_weights(n_input, n_unit, hparams), name = 'Wo')
            self.Ro = tf.Variable(init_weights(n_unit, n_unit, hparams), name = 'Ro')
            self.bo = tf.Variable(tf.zeros([1, n_unit]), name = 'bo')

    def get_new_states(self, n_state):
        new_h = tf.Variable(tf.zeros([n_state, self.n_unit]), trainable=False, name = 'h')
        new_c = tf.Variable(tf.zeros([n_state, self.n_unit]), trainable=False, name = 'c')

        return new_h, new_c


    def step(self, state, x, *d_state):
        h, c = state
        """Updates returns the state updated by input x"""
        u = tf.sigmoid(tf.matmul(x, self.Wu) + tf.matmul(h, self.Ru) + self.bu)
        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(h, self.Ri) + self.bi)
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(h, self.Rf) + self.bf)
        o = tf.sigmoid(tf.matmul(x, self.Wo) + tf.matmul(h, self.Ro) + self.bo)

        c = i*u + f*c
        h = o*tf.tanh(c)

        return h, c

    def gradient(self, error, state):
        # Using the memory state since the output state is subservient to it
        # but who knows
        _, c = state
        return tf.gradients(error, c)


class Connector(object):

    def __init__(self, n_in, n_out, params=dict()):

        with tf.name_scope('connector') as scope:
            self.Jw = tf.Variable(init_weights(n_in, n_out, params), name= 'Jw', trainable=True)

            M = np.ones([n_in, n_out])
            if 'mask' in params:
                M = params['mask']
                assert M.shape == (n_in, n_out)
            self.M = tf.constant(M.astype('float32'), name='M')

            self.signed = False
            if 'sign' in params:
                self.signed = True
                assert len(params['sign']) == n_out
                self.sign = tf.constant(np.diag(params['sign']).astype('float32'), name='sign')
                self.gain = tf.Variable([1.], name='gain', trainable=True)

    def step(self, input):
        if self.signed:
            W = tf.matmul(self.gain, tf.sigmoid(self.Jw)) * self.M
        else:
            W = self.M * self.Jw

        return tf.matmul(input, W)



