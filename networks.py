import tensorflow as tf
import layers

class Basic_Network(object):
    def __init__(self, n_input, hparams):

        n_unit = hparams['n_unit']

        if hparams['rnn_type'] == 'SRNN':
            with tf.name_scope('srnn_layer') as scope:
                self.rnn_layer = layers.SRNN_Layer(n_input, n_unit)
        elif hparams['rnn_type'] == 'LSTM':
            with tf.name_scope('lstm_layer') as scope:
                self.rnn_layer = layers.LSTM_Layer(n_input, n_unit)

        with tf.name_scope('logit_layer') as scope:
            self.logit_layer = layers.Linear_Layer(n_unit, n_input)

    def step(self, state, x_input):
        state = self.rnn_layer.step(state, x_input)
        output = self.logit_layer.output(state[0])
        return state, output

    def get_new_states(self, n_state):
        return self.rnn_layer.get_new_states(n_state)

    # Records current state of the network in storage
    def store_state_op(self, state, storage):
        return self.rnn_layer.store_state_op(state, storage)

    # This might need a more sophisticated implementation in the future
    def reset_state_op(self, state):
        reset_ops  = [state_var.assign(tf.zeros(state_var.get_shape()))
            for state_var in state]
        return tf.group(*reset_ops)
