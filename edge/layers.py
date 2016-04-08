import tensorflow as tf

def init_weights(n_input, n_unit, hparams):

    # Every ones knows this is how you do it of course
    return tf.truncated_normal([n_input, n_unit], 0.0,
        tf.sqrt(2.0)/tf.sqrt(tf.cast(n_input + n_unit, tf.float32)))



class Linear_Layer(object):
    def __init__(self, n_input, n_output, hparams):
        self.n_input = n_input
        self.n_unit = n_output

        self.W = tf.Variable(init_weights(n_input, n_output, hparams), name='W')
        self.b = tf.Variable(tf.zeros([n_output]), name='b')

    def output(self, x):
        return tf.nn.xw_plus_b(x, self.W, self.b)

class Softmax_Linear_Layer(object):
    def __init__(self, n_input, n_output, hparams):
        self.linear = Linear_Layer(n_input, n_output, hparams)

    def output(self, x):
        return tf.nn.softmax(self.linear.output(x))


class SRNN_Layer(object):
    def __init__(self, n_input, n_unit, hparams):
        self.n_input = n_input
        self.n_unit = n_unit

        self.W = tf.Variable(init_weights(n_input, n_unit, hparams), name='W')
        self.R = tf.Variable(init_weights(n_unit, n_unit, hparams), name='R')
        self.b = tf.Variable(tf.zeros([1, n_unit]), name='b')

        prediction_signals = hparams['prediction_signals']
        if 'entropy' in prediction_signals:
            self.w_ent = tf.Variable(tf.zeros([1, n_unit]), name='w_ent')
        if 'unexpected_entropy' in prediction_signals:
            self.w_unex_ent = tf.Variable(tf.zeros([1, n_unit]), name='w_unex_ent')
        if 'd_entropy' in prediction_signals:
            self.W_d_ent = tf.Variable(tf.zeros([n_unit, n_unit]),
                name='W_d_ent')
        if 'd_unexpected_entropy' in prediction_signals:
            self.W_d_unex_ent = tf.Variable(tf.zeros([n_unit, n_unit]),
                name='W_d_unex_ent')

        self.prediction_signals = prediction_signals

    def step(self, state, x):
        """Updates returns the state updated by input x"""

        h = state['h']

        u = tf.matmul(x, self.W) + tf.matmul(h, self.R) + self.b


        if 'entropy' in self.prediction_signals:
            u += tf.matmul(state['entropy'], self.w_ent)
        if 'unexpected_entropy' in self.prediction_signals:
            u += tf.matmul(state['unexpected_entropy'], self.w_unex_ent)
        if 'd_entropy' in self.prediction_signals:
            u += tf.matmul(state['d_ent'], self.W_d_ent)
        if 'd_unexpected_entropy' in self.prediction_signals:
            u += tf.matmul(state['d_unex_ent'], self.W_d_unex_ent)

        h_next = tf.tanh(u)
        state['h'] = h_next

        return h_next


class EDSRNN_Layer(SRNN_Layer):

    def __init__(self, n_input, n_unit, hparams):
        super(EDSRNN_Layer, self).__init__(n_input, n_unit, hparams)

        self.E = tf.Variable(init_weights(n_unit, n_unit, hparams), name='E')

    def step(self, h, x, *d_state):
        """Updates returns the state updated by input x"""
        dh = d_state[0]
        h = tf.sigmoid(tf.matmul(x, self.W) + tf.matmul(h, self.R)
            + tf.matmul(dh, self.E) + self.b)
        return h



class GRU_Layer(object):
    def __init__(self, n_input, n_unit, hparams):
        self.n_input = n_input
        self.n_unit = n_unit

        prediction_signals = hparams['prediction_signals']
        self.prediction_signals = prediction_signals

        # The computed update
        with tf.name_scope('update'):
            self.Wu = tf.Variable(init_weights(n_input, n_unit, hparams), name='Wu')
            self.Ru = tf.Variable(init_weights(n_unit, n_unit, hparams), name='Ru')
            self.bu = tf.Variable(tf.zeros([1, n_unit]), name='bu')

            if 'entropy' in prediction_signals:
                self.wu_ent = tf.Variable(tf.zeros([1, n_unit]), name='w_ent')
            if 'unexpected_entropy' in prediction_signals:
                self.wu_unex_ent = tf.Variable(tf.zeros([1, n_unit]), name='w_unex_ent')
            if 'd_entropy' in prediction_signals:
                self.Wu_d_ent = tf.Variable(tf.zeros([n_unit, n_unit]),
                    name='W_d_ent')
            if 'd_unexpected_entropy' in prediction_signals:
                self.Wu_d_unex_ent = tf.Variable(tf.zeros([n_unit, n_unit]),
                    name='W_d_unex_ent')

        # The reset gate
        with tf.name_scope('reset_gate'):
            self.Wr = tf.Variable(init_weights(n_input, n_unit, hparams), name='Wr')
            self.Rr = tf.Variable(init_weights(n_unit, n_unit, hparams), name='Rr')
            self.br = tf.Variable(tf.zeros([1, n_unit]), name='br')

            if 'entropy' in prediction_signals:
                self.wr_ent = tf.Variable(tf.zeros([1, n_unit]), name='w_ent')
            if 'unexpected_entropy' in prediction_signals:
                self.wr_unex_ent = tf.Variable(tf.zeros([1, n_unit]), name='w_unex_ent')
            if 'd_entropy' in prediction_signals:
                self.Wr_d_ent = tf.Variable(tf.zeros([n_unit, n_unit]),
                    name='W_d_ent')
            if 'd_unexpected_entropy' in prediction_signals:
                self.Wr_d_unex_ent = tf.Variable(tf.zeros([n_unit, n_unit]),
                    name='W_d_unex_ent')

        # The update gate
        with tf.name_scope('update_gate'):
            self.Wz = tf.Variable(init_weights(n_input, n_unit, hparams), name='Wz')
            self.Rz = tf.Variable(init_weights(n_unit, n_unit, hparams), name='Rz')
            self.bz = tf.Variable(tf.zeros([1, n_unit]), name='bz')


            if 'entropy' in prediction_signals:
                self.wz_ent = tf.Variable(tf.zeros([1, n_unit]), name='w_ent')
            if 'unexpected_entropy' in prediction_signals:
                self.wz_unex_ent = tf.Variable(tf.zeros([1, n_unit]), name='w_unex_ent')
            if 'd_entropy' in prediction_signals:
                self.Wz_d_ent = tf.Variable(tf.zeros([n_unit, n_unit]),
                    name='W_d_ent')
            if 'd_unexpected_entropy' in prediction_signals:
                self.Wz_d_unex_ent = tf.Variable(tf.zeros([n_unit, n_unit]),
                    name='W_d_unex_ent')


    def step(self, state, x):

        h = state['h']

        with tf.name_scope('reset_gate'):
            u = tf.matmul(x, self.Wr) + tf.matmul(h, self.Rr) + self.br

            if 'entropy' in self.prediction_signals:
                u += tf.matmul(state['entropy'], self.wr_ent)
            if 'unexpected_entropy' in self.prediction_signals:
                u += tf.matmul(state['unexpected_entropy'], self.wr_unex_ent)
            if 'd_entropy' in self.prediction_signals:
                u += tf.matmul(state['d_ent'], self.Wr_d_ent)
            if 'd_unexpected_entropy' in self.prediction_signals:
                u += tf.matmul(state['d_unex_ent'], self.Wr_d_unex_ent)


            r = tf.sigmoid(u)

        with tf.name_scope('update_gate'):

            u = tf.matmul(x, self.Wz) + tf.matmul(h, self.Rz) + self.bz

            if 'entropy' in self.prediction_signals:
                u += tf.matmul(state['entropy'], self.wz_ent)
            if 'unexpected_entropy' in self.prediction_signals:
                u += tf.matmul(state['unexpected_entropy'], self.wz_unex_ent)
            if 'd_entropy' in self.prediction_signals:
                u += tf.matmul(state['d_ent'], self.Wz_d_ent)
            if 'd_unexpected_entropy' in self.prediction_signals:
                u += tf.matmul(state['d_unex_ent'], self.Wz_d_unex_ent)

            z = tf.sigmoid(u)

        with tf.name_scope('update_value'):
            u = tf.matmul(x, self.Wu) + r*tf.matmul(h, self.Ru) + self.bu

            if 'entropy' in self.prediction_signals:
                u += tf.matmul(state['entropy'], self.wu_ent)
            if 'unexpected_entropy' in self.prediction_signals:
                u += tf.matmul(state['unexpected_entropy'], self.wu_unex_ent)
            if 'd_entropy' in self.prediction_signals:
                u += tf.matmul(state['d_ent'], self.Wu_d_ent)
            if 'd_unexpected_entropy' in self.prediction_signals:
                u += tf.matmul(state['d_unex_ent'], self.Wu_d_unex_ent)

            h_tilde = tf.tanh(u)


        h_next = (1.0 - z)*h + z*h_tilde
        state['h'] = h_next

        return h_next

class EDGRU_Layer(GRU_Layer):
    def __init__(self, n_input, n_unit, hparams):
        super(EDGRU_Layer, self).__init__(n_input, n_unit, hparams)

        with tf.name_scope('update'):
            self.Eu = tf.Variable(init_weights(n_unit, n_unit, hparams), name='Eu')
        with tf.name_scope('reset_gate'):
            self.Er = tf.Variable(init_weights(n_unit, n_unit, hparams), name='Er')
        with tf.name_scope('update_gate'):
            self.Ez = tf.Variable(init_weights(n_unit, n_unit, hparams), name='Ez')

    def step(self, h, x, *d_state):
        dh = d_state[0]

        # import ipdb; ipdb.set_trace()

        r = tf.sigmoid(tf.matmul(x, self.Wr) + tf.matmul(h, self.Rr)
            + tf.matmul(dh, self.Er) + self.br)
        z = tf.sigmoid(tf.matmul(x, self.Wz) + tf.matmul(h, self.Rz)
            + tf.matmul(dh, self.Ez) + self.bz)

        h_update = tf.tanh(tf.matmul(x, self.Wu) + r*(tf.matmul(h, self.Ru)
            + tf.matmul(dh, self.Eu)) + self.bu)

        h = (1.0 - z)*h + z*h_update

        return h
