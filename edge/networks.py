import tensorflow as tf
import layers

class Basic_Network(object):
    def __init__(self, n_input, hparams):

        n_unit = hparams['n_unit']

        self.uses_error = False

        if hparams['rnn_type'] == 'SRNN':
            with tf.name_scope('srnn_layer'):
                self.rnn_layer = layers.SRNN_Layer(n_input, n_unit, hparams)
        elif hparams['rnn_type'] == 'LSTM':
            with tf.name_scope('lstm_layer'):
                self.rnn_layer = layers.LSTM_Layer(n_input, n_unit, hparams)
        elif hparams['rnn_type'] == 'GRU':
            with tf.name_scope('gru_layer'):
                self.rnn_layer = layers.GRU_Layer(n_input, n_unit, hparams)
        elif hparams['rnn_type'] == 'EDSRNN':
            with tf.name_scope('edsrnn_layer'):
                self.rnn_layer = layers.EDSRNN_Layer(n_input, n_unit, hparams)
                self.uses_error = True
        elif hparams['rnn_type'] == 'EDGRU':
            with tf.name_scope('edsrnn_layer'):
                self.rnn_layer = layers.EDGRU_Layer(n_input, n_unit, hparams)
                self.uses_error = True

        with tf.name_scope('logit_layer'):
            self.logit_layer = layers.Linear_Layer(n_unit, n_input, hparams)


    def step(self, state, x_input, *d_state):
        state = self.rnn_layer.step(state, x_input, *d_state)
        output = self.logit_layer.output(state)
        return state, output

    def gradient(self, error, state):
        '''Computes the gradient of the error with respect to the network state'''
        return tf.gradients(error, state)[0]

    def get_new_states(self, n_state):
        return tf.Variable(tf.zeros([n_state, self.rnn_layer.n_unit]), trainable=False, name='h')

    # Records current state of the network in storage
    def store_state_op(self, state, storage):
        return storage.assign(state)
        # store_ops = [store_var.assign(state_var)
        #     for state_var, store_var in zip(state, storage)]
        # return tf.group(*store_ops)

    # Resets the network state to zero
    def reset_state_op(self, state):
        return state.assign(tf.zeros(state.get_shape()))
        # reset_ops  = [state_var.assign(tf.zeros(state_var.get_shape()))
        #     for state_var in state]
        # return tf.group(*reset_ops)
