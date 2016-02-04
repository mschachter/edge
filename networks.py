import tensorflow as tf

class Prediction_Network(object):
    def __init__(self, n_input, hparams):

        n_unit = hparams['n_unit']

        with tf.name_scope('lstm_layer') as scope:
            self.lstm_layer = layers.LSTM_Layer(n_input, n_unit)

        with tf.name_scope('logit_layer') as scope:
            self.logit_layer = Linear_Layer(n_unit, n_iput)

    def step(self, x):
        y = lstm_layer.step(x)
        return self.logit_layer(y)

    def set_state(self, state):
        lstm_layer.set_state(state)

    def get_state(self):
        return lstm_layer.get_state()

    def get_new_states(self, n_state):
        return lstm_layer.get_state()

    # Records current state of the network in storage
    def store_state(self, storage):
        lstm_layer.store_current_state(storage)

    def reset_state_op(self):
        return lstm_layer.reset_state()
