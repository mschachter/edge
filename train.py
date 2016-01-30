import tensorflow as tf
import yaml
import time
from datetime import datetime
import sys, os
import pprint

from input_generator import Input_Generator
import lstm
import util.text_processing as text_proc

t_run_state = time.time()

if len(sys.argv) > 1:
    param_file_path = sys.argv[1]
else:
    param_file_path = 'param/basic_text_lstm.yaml'


with open(param_file_path, 'r') as param_file:
    hparams = yaml.load(param_file)

print '-------------'
print 'Running on', param_file_path
print pprint.pprint(hparams)
print '-------------'

data_path = os.path.join(hparams['data_dir'], hparams['data_file'])
train_text, valid_text, test_text, alphabet = \
    text_proc.file_to_datasets(data_path)
n_alphabet = alphabet.size

n_batch = 64
n_prop = 10

input_generator = Input_Generator(train_text, alphabet, n_batch, n_prop)
valid_input_generator = Input_Generator(valid_text, alphabet, 1, 1)

n_unit = hparams['n_unit']

# An output layer that returns logits-- pre-softmax-- for use efficient combination
# with cross entropy
class Logit_Layer(object):
    def __init__(self, n_input, n_output):
        self.Wo = tf.Variable(tf.truncated_normal([n_input, n_output], 0.0, 0.1), name= 'Wo')
        self.bo = tf.Variable(tf.zeros([n_output]), name = 'bo')

    def logits(self, x):
        return tf.nn.xw_plus_b(x, self.Wo, self.bo)



graph = tf.Graph()
with graph.as_default():

    ## Create the graph input and state variables

    # Input nodes
    xs = [tf.placeholder(tf.float32, shape=[n_batch, n_alphabet]) for
        _ in xrange(n_prop + 1)]
    x_inputs = xs[:n_prop]
    x_labels = xs[1:]

    # The recurrent layer
    with tf.name_scope('lstm_layer') as scope:
        lstm_layer = lstm.LSTM_Layer(n_alphabet, n_unit)

    with tf.name_scope('ouput_layer') as scope:
        logit_layer = Logit_Layer(n_unit, n_alphabet)



    # Used to carry over the network state between forward propagations
    saved_output = tf.Variable(tf.zeros([n_batch, n_unit]), trainable=False, name = 'y')
    saved_state = tf.Variable(tf.zeros([n_batch, n_unit]), trainable=False, name = 'c')


    ## Build the forward propagation graph
    y = saved_output
    lstm_layer.set_state(saved_state)

    errs = list()
    for (x_input, x_label) in zip(x_inputs, x_labels):
        y = lstm_layer.step(x_input, y)
        logits = logit_layer.logits(y)
        errs.append(tf.nn.softmax_cross_entropy_with_logits(logits, x_label))
    prediction_error = tf.reduce_mean(tf.concat(0, errs))

    saved_output = y
    saved_state = lstm_layer.get_state()

    ## Build the optimizer
    with tf.name_scope('optimizer') as scope:
        t = tf.Variable(0, name= 't', trainable=False) # the step variable
        eta = tf.train.exponential_decay(10.0, t, 5000, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(eta)
        grads, params = zip(*optimizer.compute_gradients(prediction_error))
        grads, _ = tf.clip_by_global_norm(grads, 1)
        apply_grads = optimizer.apply_gradients(zip(grads, params), global_step=t)

    x_valid_input = tf.placeholder(tf.float32, shape=[1, n_alphabet])
    x_valid_label = tf.placeholder(tf.float32, shape=[1, n_alphabet])
    valid_state = ts.Variable(tf.zeros([1, n_unit]))
    valid_ouput = ts.Variable(tf.zeros([1, n_unit]))
    




num_steps = 7001 # cause why the fuck not
summary_freq = 100
mean_error = 0.0

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    #tf.train.SummaryWriter('/tmp/lstm', graph_def = session.graph_def)

    for step in xrange(num_steps):

        # Set up input value -> input var mapping
        window = input_generator.next_window()
        feed_dict = dict()
        for i in xrange(n_prop + 1):
            feed_dict[xs[i]] = window[i]

        _, error_val, eta_val = session.run(
            [apply_grads, prediction_error, eta], feed_dict=feed_dict
        )

        mean_error += error_val

        if step % summary_freq == 0 and step > 0:
            mean_error = mean_error/summary_freq
            print 'Average error at step', step, ':', mean_error, 'learning rate:', eta_val
            mean_error = 0.0
