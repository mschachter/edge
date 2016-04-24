import tensorflow as tf
import numpy as np

def init_weights(n_input, n_unit, hparams, scale=1.):

    # Every ones knows this is how you do it of course
    std = scale*(tf.sqrt(2.0) / tf.sqrt(tf.cast(n_input + n_unit, tf.float32)))
    return tf.truncated_normal([n_input, n_unit], 0.0, std)

def init_weights_ei(n_input, n_unit, hparams, scale=1.):
    # A sigmoid is applied to the weights generated here. To keep the network from blowing up,
    # we need the weights to be negative, so the output of the sigmoid is close to zero.
    std = scale*(tf.sqrt(2.0) / tf.sqrt(tf.cast(n_input + n_unit, tf.float32)))
    return -tf.abs(tf.truncated_normal([n_input, n_unit], 0.0, std))


class Linear_Layer(object):
    def __init__(self, n_input, n_output, hparams):
        self.n_input = n_input
        self.n_unit = n_output

        self.W = tf.Variable(init_weights(n_input, n_output, hparams), name= 'W')
        self.b = tf.Variable(tf.zeros([n_output]), name = 'b')

    def output(self, x):
        v = tf.nn.xw_plus_b(x, self.W, self.b)
        return v


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
            assert hparams['activation'] in ['sigmoid', 'tanh', 'relu', 'elu']
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

    def weight_cost(self):
        if 'lambda2' in self.hparams and self.hparams['lambda2'] > 0:
            l2_W = tf.reduce_mean(tf.square(self.W))
            l2_R = tf.reduce_mean(tf.square(self.R))
            l2_b = tf.reduce_mean(tf.square(self.b))
            return self.hparams['lambda2']*(l2_W + l2_R + l2_b)

    def gradient(self, error, state):
        return tf.gradients(error, state[0])


class EI_Layer(object):

    def __init__(self, n_input, n_unit, hparams):
        self.n_input = n_input
        self.n_unit = n_unit

        assert 'sign' in hparams, "Must supply 'sign' in hparams, a vector of length n_unit that has 1 for excitatory neurons, -1 for inhibitory neurons"
        self.sign = hparams['sign']
        assert len(self.sign) == self.n_unit, "Wrong size for hparams['sign'], must be of length %d" % n_unit

        M = np.ones([n_unit, n_unit])
        if 'mask' in hparams:
            M = hparams['mask']
        assert M.shape == (n_unit, n_unit)

        with tf.name_scope('ei_layer'):
            self.M = tf.constant(M.astype('float32'), name='M')
            self.D = tf.constant(np.diag(self.sign.astype('float32')), name='D')
            self.gr = tf.Variable(tf.ones([1]), name='gr', trainable=True)
            self.Jr = tf.Variable(init_weights_ei(n_unit, n_unit, hparams, scale=1000.), name='Jr', trainable=True)
            self.b = tf.Variable(tf.zeros([1, n_unit]), name='b', trainable=True)

            self.R = tf.matmul(self.D, tf.sigmoid(self.Jr)) * self.M * self.gr

            self.W = tf.Variable(init_weights(n_input, n_unit, hparams), name='W', trainable=True)

        if 'activation' in hparams:
            assert hparams['activation'] in ['sigmoid', 'relu', 'elu']
            self.activation = getattr(tf.nn, hparams['activation'])
        else:
            self.activation = tf.nn.relu

    def get_new_states(self, n_state):
        new_h = tf.Variable(tf.zeros([n_state, self.n_unit]), trainable=False, name='h')
        return new_h,

    def step(self, state, x, *d_state, **kwargs):
        """Updates returns the state updated by input x"""
        h = state[0]

        xxx = tf.matmul(x, self.W)
        hhh = tf.matmul(h, self.R)
        h = self.activation(xxx + hhh + self.b)

        return h,

    def gradient(self, error, state):
        return tf.gradients(error, state[0])

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



