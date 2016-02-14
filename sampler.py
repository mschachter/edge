import tensorflow as tf
import numpy as np

def sample_dist(dist):
    r = np.random.rand()
    return np.where(np.cumsum(dist) > r)[0][0]

class Sampler(object):

    def __init__(self, net, alphabet):

        self.net = net
        self.alphabet = alphabet


        with tf.name_scope('sampler') as scope:
            cur_state = net.get_new_states(1)
            cur_d_state = net.get_new_states(1)

            self.input = tf.placeholder(tf.float32, shape=[1, len(alphabet)])
            self.output = tf.placeholder(tf.float32, shape=[1, len(alphabet)])

            next_state, logits = net.step(cur_state, self.sample_input, cur_d_state)
            self.prediction = tf.nn.softmax(logits)
            err = tf.nn.softmax_cross_entropy_with_logits(logits, self.output)

            next_d_state = net.gradient(err, next_state)

            self.store_state = net.store_state_op(next_state, cur_state)
            self.store_d_state = net.store_state_op(next_d_state, cur_d_state)

            self.reset_state = net.reset_state_op(cur_state)
            self.reset_d_state = net.reset_state_op(cur_d_state)

    def reset(self):
        session.run(self.reset_sample_state)
        if net.uses_error:
            session.run(self.reset_d_state)

    def predict_next(self, sample_input):
        prediction, _ = session.run([self.prediction, self.store_state],
            feed_dict = {self.input = sample_input})
        return prediction

    def sample(self, session, prime='Alice was ', n_sample = 100, bias = 0.0):

        session.run(self.reset_sample_state)
        if net.uses_error:
            session.run(self.reset_d_state)

        bias = np.array([bias])
        n_alpha = len(self.alphabet)

        for i in xrange(len(prime)):
            alpha_id = np.where(self.alphabet == prime[i])
            input_val = np.zeros([1, n_alpha], dtype=np.float32)
            input_val[0, alpha_id] = 1
            to_compute = [self.prediction, self.store_state]
            prediction, _ = session.run([self.prediction, self.store_state],
                feed_dict = {self.input: input_val, self.bias: bias})

            if net.uses

        sample_string = ''
        for i in xrange(n_sample):
            alpha_id = sample_dist(prediction)
            sample_string += self.alphabet[alpha_id]

            input_val = np.zeros([1, n_alpha], dtype=np.float32)
            input_val[0, alpha_id] = 1

            prediction, _ = session.run([self.prediction, self.store_state],
                feed_dict = {self.input: input_val, self.bias: bias})

        return prime, sample_string
