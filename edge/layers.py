import tensorflow as tf

class Linear_Layer(object):
    def __init__(self, n_input, n_output):
        self.W = tf.Variable(tf.truncated_normal([n_input, n_output], 0.0, 0.1), name= 'W')
        self.b = tf.Variable(tf.zeros([n_output]), name = 'b')

    def output(self, x):
        return tf.nn.xw_plus_b(x, self.W, self.b)


class SRNN_Layer(object):

    def __init__(self, n_input, n_unit):
        self.n_input = n_input
        self.n_unit = n_unit

        self.W = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1), name = 'W')
        self.R = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'R')
        self.b = tf.Variable(tf.zeros([1, n_unit]), name = 'b')

    def get_new_states(self, n_state):
        new_h = tf.Variable(tf.zeros([n_state, self.n_unit]), trainable=False, name = 'h')
        return new_h,

    def step(self, state, x, *d_state):
        """Updates returns the state updated by input x"""
        h = state[0]
        h = tf.tanh(tf.matmul(x, self.W) + tf.matmul(h, self.R) + self.b)

        return h,

    def gradient(self, error, state):
        return tf.gradients(error, state[0])


class EDSRNN_Layer(SRNN_Layer):

    def __init__(self, n_input, n_unit):
        super(EDSRNN_Layer, self).__init__(n_input, n_unit)

        self.E = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'E')

    def step(self, state, x, *d_state):
        """Updates returns the state updated by input x"""
        d_state, = d_state
        state = tf.sigmoid(tf.matmul(x, self.W) + tf.matmul(state[0], self.R)
            + tf.matmul(d_state[0], self.E) + self.b)

        return state,

class GRU_Layer(object):
    def __init__(self, n_input, n_unit):
        self.n_input = n_input
        self.n_unit = n_unit

        # The computed update
        with tf.name_scope('update') as scope:
            self.Wu = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1), name = 'Wu')
            self.Ru = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'Ru')
            self.bu = tf.Variable(tf.zeros([1, n_unit]), name = 'bu')

        # The reset gate
        with tf.name_scope('reset_gate') as scope:
            self.Wr = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1), name = 'Wr')
            self.Rr = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'Rr')
            self.br = tf.Variable(tf.zeros([1, n_unit]), name = 'br')

        # The update gate
        with tf.name_scope('update_gate') as scope:
            self.Wz = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1), name = 'Wz')
            self.Rz = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'Rz')
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

class LSTM_Layer(object):


    def __init__(self, n_input, n_unit):
        self.n_input = n_input
        self.n_unit = n_unit

        ## Layer parameters: W input weights, R recurrent weights, b bias

        # The computed update
        with tf.name_scope('update') as scope:
            self.Wu = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1), name = 'Wu')
            self.Ru = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'Ru')
            self.bu = tf.Variable(tf.zeros([1, n_unit]), name = 'bu')

        # Input gate
        with tf.name_scope('input_gate') as scope:
            self.Wi = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1), name = 'Wi')
            self.Ri = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'Ri')
            self.bi = tf.Variable(tf.zeros([1, n_unit]), name = 'bi')

        # Forget gate
        with tf.name_scope('forget_gate') as scope:
            self.Wf = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1), name = 'Wf')
            self.Rf = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'Rf')
            # using a positive bias as suggested in Joxefowicx 2015
            self.bf = tf.Variable(tf.ones([1, n_unit]), name = 'bf')

        # Output gate
        with tf.name_scope('output_gate') as scope:
            self.Wo = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1), name = 'Wo')
            self.Ro = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'Ro')
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
