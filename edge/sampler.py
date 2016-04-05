import tensorflow as tf
import numpy as np
import util.text_processing as textproc



def sample_dist(dist, bias):
    logits = np.log(dist)
    biased_dist = np.exp((1 + bias)*logits)
    biased_dist = biased_dist/np.sum(biased_dist)

    r = np.random.rand()
    return np.where(np.cumsum(dist) >= r)[0][0]


class Sampler(object):

    def __init__(self, net, alphabet):

        self.net = net
        self.alphabet = alphabet


        with tf.name_scope('sampler'):
            state_store = net.get_new_state_store(1)

            state = net.state_from_store(state_store)
            self.x_cur = tf.placeholder(tf.float32, shape=[1, len(alphabet)])
            self.prediction = net.step(state, self.x_cur)
            self.store_post_pred = net.store_state_op(state, state_store)



            state = net.state_from_store(state_store)
            self.x_pred = tf.placeholder(tf.float32, shape=[1, len(alphabet)])
            self.x_next = tf.placeholder(tf.float32, shape=[1, len(alphabet)])
            self.xent, self.ent = net.evaluate_prediction(state, self.x_pred, self.x_next)
            self.store_post_eval = net.store_state_op(state, state_store)



            self.reset_state = net.reset_state_op(state_store)

    def predict(self, session, cur_x):
        prediction = session.run([self.prediction, self.store_post_pred],
            feed_dict = {self.x_cur: cur_x})[0]
        return prediction

    def evaluate_prediction(self, session, pred_x, next_x):
        feed = {self.x_next: next_x, self.x_pred: pred_x}
        xent = session.run([self.xent, self.store_post_eval], feed)[0][0][0]
        return xent

    def sample(self, session, prime='alice was ', n_sample = 500, bias = 0.0):

        session.run(self.reset_state)

        bias = np.array([bias])

        # Prime the network
        cur_x = textproc.char_to_onehot(prime[0], self.alphabet)
        for i in range(len(prime) - 1):
            prediction = self.predict(session, cur_x)
            next_x = textproc.char_to_onehot(prime[i+1], self.alphabet)
            # Err needs to be computed even though not used in case
            # the network is error dynamic
            self.evaluate_prediction(session, prediction, next_x)
            cur_x = next_x

        # Sample new inputs
        sample_string = ''
        for i in xrange(n_sample):
            prediction = self.predict(session, cur_x)
            alpha_id = sample_dist(prediction, bias)
            sample_string += self.alphabet[alpha_id]

            next_x = textproc.id_to_onehot(alpha_id, self.alphabet)
            self.evaluate_prediction(session, prediction, next_x)
            cur_x = next_x

        return prime, sample_string

    def test_prediction_error(self, session, test_text):
        session.run(self.reset_state)

        cur_x = textproc.id_to_onehot(test_text[0], self.alphabet)
        mean_err = 0.0
        for i in range(len(test_text) - 1):
            prediction = self.predict(session, cur_x)
            next_x = textproc.id_to_onehot(test_text[i+1], self.alphabet)
            mean_err += self.evaluate_prediction(session, prediction, next_x)
            cur_x = next_x

        return mean_err/(len(test_text)  -1)
