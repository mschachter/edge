import copy

import tensorflow as tf
import layers


def entropy(p):
    eps  = 1e-13
    p_safe = tf.maximum(p, eps)
    return -tf.reduce_sum(p_safe*tf.log(p_safe), 1, keep_dims=True)

def cross_entropy(p, p_true):
    eps = 1e-13
    p_safe = tf.maximum(p, eps)
    return -tf.reduce_sum(p_true*tf.log(p_safe), 1, keep_dims=True)

class Prediction_Network(object):
    def __init__(self, hparams):

        self.n_input = hparams['n_alphabet']

        self.prediction_signals = hparams['prediction_signals']

        self.rnn_layers = []

        # create one or more rnn layers
        layer_inputs = self.n_input
        for i, n_unit in enumerate(hparams['rnn_layers']):
            if hparams['rnn_type'] == 'SRNN':
                with tf.name_scope('srnn_layer' + str(i)):
                    layer =  layers.SRNN_Layer(layer_inputs, n_unit, hparams)
            elif hparams['rnn_type'] == 'GRU':
                with tf.name_scope('gru_layer' + str(i)):
                    layer = layers.GRU_Layer(layer_inputs, n_unit, hparams)
            self.rnn_layers.append(layer)
            layer_inputs = n_unit

        # create an ouput layer
        with tf.name_scope('ouput_layer'):
            self.prediction_layer = layers.Softmax_Prediction_Layer(layer_inputs,
                self.n_input, hparams)




    # Note: this acts by side effects on the state
    def step(self, state, x):
        layer_input = x

        for i, rnn_layer in enumerate(self.rnn_layers):
            layer_state = state[i]
            layer_input = rnn_layer.step(layer_state, layer_input)

        prediction = self.prediction_layer.output(layer_input)

        return prediction

    def evaluate_prediction(self, state, prediction, x):
        ent = entropy(prediction)
        cross_ent = cross_entropy(prediction, x)
        excess_ent = cross_ent

        for i in range(len(self.rnn_layers)):
            layer_state = state[i]
            if 'entropy' in self.prediction_signals:
                layer_state['entropy'] = ent
            if 'excess_entropy' in self.prediction_signals:
                layer_state['excess_entropy'] = excess_ent
            if 'd_entropy' in self.prediction_signals:
                layer_state['d_ent'] = tf.gradients(entropy, layer_state['h'])[0]
            if 'd_excess_entropy' in self.prediction_signals:
                layer_state['d_ex_ent'] = tf.gradients(excess_ent, layer_state['h'])[0]

        return cross_ent, ent

    def get_new_state_store(self, n_state):
        state_store = []
        for rnn_layer in self.rnn_layers:
            layer_store = {}

            layer_store['h'] = tf.Variable(tf.zeros([n_state, rnn_layer.n_unit]),
                trainable=False, name='h')

            if 'entropy' in self.prediction_signals:
                layer_store['entropy'] = tf.Variable(tf.zeros([n_state, 1]),
                    name = 'entropy')
            if 'excess_entropy' in self.prediction_signals:
                layer_store['excess_entropy'] = tf.Variable(tf.zeros([n_state, 1]),
                    name = 'excess_entropy')
            if 'd_entropy' in self.prediction_signals:
                layer_store['d_ent'] = tf.Variable(tf.zeros([n_state, rnn_layer.n_unit]),
                    trainable=False, name='d_ent')
            if 'd_excess_entropy' in self.prediction_signals:
                layer_store['d_ex_ent'] = tf.Variable(tf.zeros([n_state, rnn_layer.n_unit]),
                    trainable=False, name='d_ex_ext')

            state_store.append(layer_store)

        return state_store

    def state_from_store(self, state_store):
        return [copy.copy(layer_store) for layer_store in state_store]

    # Records current state of the network in storage
    def store_state_op(self, state, state_store):
        store_ops = []
        for i in range(len(self.rnn_layers)):
            layer_state = state[i]
            layer_storage = state_store[i]

            for key in layer_state.keys():
                store_op = layer_storage[key].assign(layer_state[key])
                store_ops.append(store_op)

        return tf.group(*store_ops)


    # Resets the network state to zero
    def reset_state_op(self, state):
        reset_ops = []
        for i in range(len(self.rnn_layers)):
            layer_state = state[i]

            for key in layer_state.keys():
                state_var = layer_state[key]
                reset_op = state_var.assign(tf.zeros(state_var.get_shape()))
                reset_ops.append(reset_op)

        return tf.group(*reset_ops)
