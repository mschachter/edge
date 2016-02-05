import tensorflow as tf

class Linear_Layer(object):
    def __init__(self, n_input, n_output):
        self.Wo = tf.Variable(tf.truncated_normal([n_input, n_output], 0.0, 0.1), name= 'W')
        self.bo = tf.Variable(tf.zeros([n_output]), name = 'b')

    def output(self, x):
        return tf.nn.xw_plus_b(x, self.Wo, self.bo)

# 
# class SRNN_Layer(object):
#
#     def __init__(self, n_input, n_unit):
#         self.n_input = n_input
#         self.n_unit = n_unit
#
#         self.W = tf.Variable(tf.truncated_normal([n_input, n_unit], 0.0, 0.1), name = 'W')
#         self.R = tf.Variable(tf.truncated_normal([n_unit, n_unit], 0.0, 0.1), name = 'R')
#         self.b = tf.Variable(tf.zeros([1, n_unit]), name = 'b')
#
#         self.y = None
#



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

        # The activity and memory state
        self.y = None
        self.c = None

    def get_new_states(self, n_state):
        new_y = tf.Variable(tf.zeros([n_state, self.n_unit]), trainable=False, name = 'y')
        new_c = tf.Variable(tf.zeros([n_state, self.n_unit]), trainable=False, name = 'c')

        return new_y, new_c

    def set_state(self, state):
        self.y, self.c = state

    def get_state(self):
        return self.y, self.c

    # Returns an op that stores the current network state in storage
    def store_state_op(self, storage):
        y_storage, c_storage = storage
        return tf.group(y_storage.assign(self.y), c_storage.assign(self.c))


    def step(self, x):
        """Updates the internal memory state and returns the output"""

        # import ipdb
        # ipdb.set_trace()

        assert self.c is not None and self.y is not None # need to set the state externally before stepping

        u = tf.sigmoid(tf.matmul(x, self.Wu) + tf.matmul(self.y, self.Ru) + self.bu)
        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(self.y, self.Ri) + self.bi)
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(self.y, self.Rf) + self.bf)

        self.c = i*u + f*self.c
        self.y = tf.tanh(self.c)

        return self.y
