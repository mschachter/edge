import tensorflow as tf
import layers

class Prediction_Network(object):
    def __init__(self, n_input, hparams):

        n_unit = hparams['n_unit']

        with tf.name_scope('lstm_layer') as scope:
            self.lstm_layer = layers.LSTM_Layer(n_input, n_unit)

        with tf.name_scope('logit_layer') as scope:
            self.logit_layer = layers.Linear_Layer(n_unit, n_input)

    def step(self, x):
        y = self.lstm_layer.step(x)
        return self.logit_layer.output(y)

    def set_state(self, state):
        self.lstm_layer.set_state(state)

    def get_state(self):
        return self.lstm_layer.get_state()

    def get_new_states(self, n_state):
        return self.lstm_layer.get_new_states(n_state)

    # Records current state of the network in storage
    def store_state_op(self, storage):
        return self.lstm_layer.store_state_op(storage)
