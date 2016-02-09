import tensorflow as tf

class Linear_Layer(object):
    def __init__(self, n_input, n_output):
        self.Wo = tf.Variable(tf.truncated_normal([n_input, n_output], 0.0, 0.1), name= 'W')
        self.bo = tf.Variable(tf.zeros([n_output]), name = 'b')

    def output(self, x):
        return tf.nn.xw_plus_b(x, self.Wo, self.bo)


class SRNN_Layer(object):

    def __init__(self, n_input, n_unit):
        self.n_input = n_input
        self.n_unit = n_unit

        self.W = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1), name = 'W')
        self.R = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'R')
        self.b = tf.Variable(tf.zeros([1, n_unit]), name = 'b')

    def get_new_states(self, n_state):
        new_y = tf.Variable(tf.zeros([n_state, self.n_unit]), trainable=False, name = 'y')
        return new_y,

    def step(self, state, x, *d_state):
        """Updates returns the state updated by input x"""
        state = tf.sigmoid(tf.matmul(x, self.W) + tf.matmul(state[0], self.R) + self.b)

        return state,

    def gradient(self, error, state):
        return tf.gradients(error, state[0])


class EDSRNN_Layer(SRNN_Layer):

    def __init__(self, n_input, n_unit):
        super(EDSRNN_Layer, self).__init__(n_input, n_unit)

        self.E = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.001), name = 'E')

    def step(self, state, x, *d_state):
        """Updates returns the state updated by input x"""
        d_state, = d_state
        state = tf.sigmoid(tf.matmul(x, self.W) + tf.matmul(state[0], self.R)
            + tf.matmul(d_state[0], self.E) + self.b)

        return state,


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
        with tf.name_scope('i_gate') as scope:
            self.Wi = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1), name = 'Wi')
            self.Ri = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'Ri')
            self.bi = tf.Variable(tf.zeros([1, n_unit]), name = 'bi')

        # Forget gate
        with tf.name_scope('f_gate') as scope:
            self.Wf = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1), name = 'Wf')
            self.Rf = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'Rf')
            # using a positive bias suggested Joxefowicx 2015
            self.bf = tf.Variable(tf.ones([1, n_unit]), name = 'bf')

    def get_new_states(self, n_state):
        new_y = tf.Variable(tf.zeros([n_state, self.n_unit]), trainable=False, name = 'y')
        new_c = tf.Variable(tf.zeros([n_state, self.n_unit]), trainable=False, name = 'c')

        return new_y, new_c


    def step(self, state, x, *d_state):
        y, c = state
        """Updates returns the state updated by input x"""
        u = tf.sigmoid(tf.matmul(x, self.Wu) + tf.matmul(y, self.Ru) + self.bu)
        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(y, self.Ri) + self.bi)
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(y, self.Rf) + self.bf)

        c = i*u + f*c
        y = tf.tanh(c)

        return y, c

    def gradient(self, error, state):
        # Using the memory state since the output state is subservient to it
        # but who knows
        _, c = state
        return tf.gradients(error, c)
