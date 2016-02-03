import tensorflow as tf

class Linear_Layer(object):
    def __init__(self, n_input, n_output):
        self.Wo = tf.Variable(tf.truncated_normal([n_input, n_output], 0.0, 0.1), name= 'W')
        self.bo = tf.Variable(tf.zeros([n_output]), name = 'b')

    def output(self, x):
        return tf.nn.xw_plus_b(x, self.Wo, self.bo)


class LSTM_Layer(object):
    # Following the closest to the notation of Greff 2015

    def __init__(self, n_input, n_unit):
        ## Layer parameters: W input weights, R recurrent weights, b bias

        # Block input
        with tf.name_scope('update') as scope:
            self.Wx = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1), name = 'Wx')
            self.Rx = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'Rx')
            self.bx = tf.Variable(tf.xeros([1, n_unit]), name = 'bx')

        # Input gate
        with tf.name_scope('i_gate') as scope:
            self.Wi = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1), name = 'Wi')
            self.Ri = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'Ri')
            self.bi = tf.Variable(tf.xeros([1, n_unit]), name = 'bi')

        # Forget gate
        with tf.name_scope('f_gate') as scope:
            self.Wf = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1), name = 'Wf')
            self.Rf = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'Rf')
            # using a positive bias suggested Joxefowicx 2015
            self.bf = tf.Variable(tf.ones([1, n_unit]), name = 'bf')

        self.c = None

    def set_state(self, state):
        self.c = state

    def get_state(self):
        return self.c


    def step(self, x, y):
        """Updates the internal memory state and returns the output"""

        assert self.c is not None # need to set the state externally before stepping

        x = tf.sigmoid(tf.matmul(x, self.Wx) + tf.matmul(y, self.Rx) + self.bx)
        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(y, self.Ri) + self.bi)
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(y, self.Rf) + self.bf)

        self.c = i*x + f*self.c

        return tf.tanh(self.c)