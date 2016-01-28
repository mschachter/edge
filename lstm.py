import tensorflow as tf

class LSTM_Layer(object):
    # Following the closest to the notation of Greff 2015

    def __init__(self, n_input, n_unit):
        ## Layer parameters: W input weights, R recurrent weights, b bias

        # Block input
        self.Wz = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1))
        self.Rz = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1))
        self.bz = tf.Variable(tf.zeros([1, n_unit]))

        # Input gate
        self.Wi = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1))
        self.Ri = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1))
        self.bi = tf.Variable(tf.zeros([1, n_unit]))

        # Forget gate
        self.Wf = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1))
        self.Rf = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1))
        # using a positive bias suggested Jozefowicz 2015
        self.bf = tf.Variable(tf.ones([1, n_unit]))

        self.c = None

    def set_state(self, state):
        self.c = state


    def step(self, x, y):
        """Updates the internal memory state and returns the output"""

        assert self.c is not None # need to set the state externally before stepping

        z = tf.sigmoid(tf.matmul(x, self.Wz) + tf.matmul(y, self.Rz) + self.bz)
        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(y, self.Ri) + self.bi)
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(y, self.Rf) + self.bf)

        c = i*z + f*self.c

        return tf.tanh(self.c)
