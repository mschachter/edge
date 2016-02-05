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

            self.input_var = tf.placeholder(tf.float32, shape=[1, len(alphabet)])
            self.bias_var = tf.placeholder(tf.float32, shape=[1])
            next_state, output = net.step(cur_state, self.input_var)
            self.prediction = tf.nn.softmax(output*(1.0 + self.bias_var))

            self.store_sample_state = net.store_state_op(next_state, cur_state)

            self.reset_sample_state = net.reset_state_op(cur_state)

    def sample(self, session, prime='Alice was ', n_sample = 100, bias = 0.0):

        session.run(self.reset_sample_state)

        bias = np.array([bias])
        n_alpha = len(self.alphabet)

        for i in xrange(len(prime)):
            alpha_id = np.where(self.alphabet == prime[i])
            input_val = np.zeros([1, n_alpha], dtype=np.float32)
            input_val[0, alpha_id] = 1
            prediction, _ = session.run([self.prediction, self.store_sample_state],
                feed_dict = {self.input_var: input_val, self.bias_var: bias})

        sample_string = ''
        for i in xrange(n_sample):
            alpha_id = sample_dist(prediction)
            sample_string += self.alphabet[alpha_id]

            input_val = np.zeros([1, n_alpha], dtype=np.float32)
            input_val[0, alpha_id] = 1

            prediction, _ = session.run([self.prediction, self.store_sample_state],
                feed_dict = {self.input_var: input_val, self.bias_var: bias})

        return prime, sample_string
