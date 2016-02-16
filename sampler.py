import tensorflow as tf
import numpy as np
import util.text_processing as textproc

def sample_dist(dist, bias):
    logits = np.log(dist)
    dist = np.exp((1 + bias)*logits)
    dist /= np.sum(dist)

    r = np.random.rand()
    return np.where(np.cumsum(dist) >= r)[0][0]



class Sampler(object):

    def __init__(self, net, alphabet):

        self.net = net
        self.alphabet = alphabet


        with tf.name_scope('sampler'):
            self.cur_state = net.get_new_states(1)
            cur_d_state = net.get_new_states(1)

            self.input = tf.placeholder(tf.float32, shape=[1, len(alphabet)])
            self.next_input = tf.placeholder(tf.float32, shape=[1, len(alphabet)])

            next_state, logits = net.step(self.cur_state, self.input, cur_d_state)
            self.next_state = next_state
            self.prediction = tf.nn.softmax(logits)
            self.err = tf.nn.softmax_cross_entropy_with_logits(logits, self.next_input)

            next_d_state = net.gradient(self.err, next_state)

            self.store_state = net.store_state_op(next_state, self.cur_state)
            self.store_d_state = net.store_state_op(next_d_state, cur_d_state)

            self.reset_state = net.reset_state_op(self.cur_state)
            self.reset_d_state = net.reset_state_op(cur_d_state)

    def reset(self, session):
        session.run(self.reset_sample_state)
        if self.net.uses_error:
            session.run(self.reset_d_state)

    def predict_next(self, session, cur_input):
        prediction, _ = session.run([self.prediction, self.store_state],
            feed_dict = {self.input: cur_input})
        return prediction

    def compute_prediction_error(self, session, next_input):
        to_compute = [self.err]
        if self.net.uses_error:
            to_compute.append(self.store_d_state)
        #import ipdb; ipdb.set_trace()
        cur_state_val = self.cur_state[0].eval(session)
        # TODO: handle state mapping for LSTM state
        err = session.run(to_compute, feed_dict = {self.next_input: next_input,
            self.next_state[0]: cur_state_val})[0]
        return err

    def sample(self, session, prime='alice was ', n_sample = 100, bias = 0.0):

        session.run(self.reset_state)
        if self.net.uses_error:
            session.run(self.reset_d_state)

        bias = np.array([bias])

        # Prime the network
        cur_input = textproc.char_to_onehot(prime[0], self.alphabet)
        for i in range(len(prime) - 1):
            prediction = self.predict_next(session, cur_input)
            next_input = textproc.char_to_onehot(prime[i+1], self.alphabet)
            # Err needs to be computed even though not used in case
            # the network is error dynamic
            self.compute_prediction_error(session, next_input)

            cur_input = next_input

        # Sample new inputs
        sample_string = ''
        for i in xrange(n_sample):
            prediction = self.predict_next(session, cur_input)

            alpha_id = sample_dist(prediction, bias)
            sample_string += self.alphabet[alpha_id]
            next_input = textproc.id_to_onehot(alpha_id, self.alphabet)

            self.compute_prediction_error(session, next_input)

            cur_input = next_input

        return prime, sample_string

    def test_prediction_error(self, session, test_text):
        session.run(self.reset_state)
        if self.net.uses_error:
            session.run(self.reset_d_state)

        cur_input = textproc.id_to_onehot(test_text[0], self.alphabet)
        mean_err = 0.0
        for i in range(len(test_text) - 1):
            self.predict_next(session, cur_input)
            next_input = textproc.id_to_onehot(test_text[i+1], self.alphabet)
            mean_err += self.compute_prediction_error(session, next_input)[0]
            cur_input = next_input

        return mean_err/(len(test_text)  -1)
