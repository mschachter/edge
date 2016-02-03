import tensorflow as tf
import yaml
import time
from datetime import datetime
import sys, os
import pprint

from input_generator import Input_Generator
import lstm
import util.text_processing as text_proc
from networks import Prediction_Network

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

n_train = len(train_text)
n_valid = len(valid_text)

n_batch = 64
n_prop = 10

input_generator = Input_Generator(train_text, alphabet, n_batch, n_prop)
valid_input_generator = Input_Generator(valid_text, alphabet, 1, 1)

n_unit = hparams['n_unit']




graph = tf.Graph()
with graph.as_default():


    ## Define the network
    net = Prediction_Network(n_alphabet, hparams)

    ## Create the training graph
    # Input nodes
    xs = [tf.placeholder(tf.float32, shape=[n_batch, n_alphabet]) for
        _ in xrange(n_prop + 1)]
    x_inputs = xs[:-1]
    x_labels = xs[1:]

    # Used to carry over the network state between forward propagations
    saved_output = tf.Variable(tf.zeros([n_batch, n_unit]), trainable=False, name = 'y')
    saved_state = tf.Variable(tf.zeros([n_batch, n_unit]), trainable=False, name = 'c')
    y = saved_output
    net.set_state(saved_state)

    # Forward propagation
    errs = list()
    for (x_input, x_label) in zip(x_inputs, x_labels):
        y, logits = net.step(x_input, y)
        errs.append(tf.nn.softmax_cross_entropy_with_logits(logits, x_label))
    prediction_error = tf.reduce_mean(tf.concat(0, errs))
    saved_output = y
    saved_state = net.get_state()

    ## Build the optimizer
    with tf.name_scope('optimizer') as scope:
        t = tf.Variable(0, name= 't', trainable=False) # the step variable
        eta = tf.train.exponential_decay(10.0, t, 5000, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(eta)
        grads, params = zip(*optimizer.compute_gradients(prediction_error))
        grads, _ = tf.clip_by_global_norm(grads, 1)
        apply_grads = optimizer.apply_gradients(zip(grads, params), global_step=t)

    valid_x_input = tf.placeholder(tf.float32, shape=[1, n_alphabet])
    valid_x_label = tf.placeholder(tf.float32, shape=[1, n_alphabet])
    valid_state = ts.Variable(tf.zeros([1, n_unit]), trainable=False)
    valid_output = ts.Variable(tf.zeros([1, n_unit]), trainable=False)

    net.set_state(valid_state)




num_steps = 7001 # cause why the fuck not
summary_freq = 100
mean_error = 0.0

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    #tf.train.SummaryWriter('/tmp/lstm', graph_def = session.graph_def)

    for step in range(num_steps):

        # Set up input value -> input var mapping
        window = input_generator.next_window()
        feed_dict = dict()
        for i in range(n_prop + 1):
            feed_dict[xs[i]] = window[i]

        _, error_val, eta_val = session.run(
            [apply_grads, prediction_error, eta], feed_dict=feed_dict
        )

        mean_error += error_val

        if step % summary_freq == 0 and step > 0:
            mean_error = mean_error/summary_freq

            valid_state.assign(tf.zeros([1, n_unit]))

            valid_output.assign(tf.zeros([1, n_unit]))
            valid_errs = list()
            for valid_idx in range(n_valid):
                net.step()

            print 'Average error at step', step, ':', mean_error, 'learning rate:', eta_val
            mean_error = 0.0
