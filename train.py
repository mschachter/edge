import tensorflow as tf
import InputGenerator
import lstm
import yaml
import time
from datetime import datetime
import sys

t_run_state = time.time()

if len(sys.argv) > 1:
    param_file_path = sys.argv[1]
else:
    param_file_path = 'params/basic_text_lstm.yaml'

print 'Running on', param_file_path

with open(param_file_path, 'r') as param_file:
    hparams = yaml.load(param_file)

print '-------------'
print hparams
print '-------------'

####################### TODO: where I left off

input_generator = Input_Generator(train_text, batch_size, n_bptt)
n_alphabet = input_generator.n_alphabet

n_unit = 64

graph = tf.Graph()
with graph.as_default():

    ## Create the graph input and state variables

    # Input nodes
    xs = [tf.placeholder(tf.float32, shape=[batch_size, n_alphabet]) for
        _ in xrange(n_bptt + 1)]
    x_inputs = xs[:n_bptt]
    x_labels = xs[1:]

    lstm_layer = lstm.LSTM_Layer(num_nodes)

    # Ouput weights and biases
    Wo = tf.Variable(tf.truncated_normal([n_unit, n_alphabet], 0.0, 0.1))
    bo = tf.Variable(tf.zeros([n_alphabet]))

    # Used to carry over the network state between forward propagations
    saved_y = tf.Variable(tf.zeros([batch_size, n_unit]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, n_unit]), trainable=False)


    ## Build the forward propagation graph
    y = saved_y
    lstm_layer.set_state(saved_state)

    ys = list()
    for x in x_inputs:
        y = lstm_layer.step(x, y)
        ys.append(y)

    with tf.control_dependencies([saved_y.assign(y),
        save_state.assign(lstm_layer.c)]):

        x_predictions = tf.nn.xw_plus_b(tf.concat(0, ys), Wo, bo)
        prediction_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            x_predictions, tf.concat(0, x_labels)
        ))


    ## Build the optimizer
    t = tf.Variable(0) # the step variable
    eta = tf.train.exponential_decay(10.0, t, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(eta)
    grads, params = zip(*optimizer.compute_gradients(prediction_error))
    grads, _ = tf.clip_by_global_norm(grads, 1)
    apply_grads = optimizer.apply_gradients(zip(grads, params), global_step=t)

num_steps = 7001 # cause why the fuck not
summary_freq = 100
mean_error = 0.0

with tf.Session(graph=graph):
    tf.initialize_all_variables().run()

    for step in xrange(num_steps):

        # Set up input value -> input var mapping
        batches = input_generator.next_batches()
        feed_dict = dict()
        for i in xrange(n_bptt + 1):
            feed_dict[xs[i]] = batches[i]

        _, error_val, eta_val = session.run(
            [apply_grads, prediction_error, eta], feed_dict=feed_dict
        )
        mean_error += error_val

        if step % summary_frequency == 0 && step > 0:
            mean_error = mean_error/summary_freq
            print 'Average error at step', step, ':', mean_error, 'learning rate:', eta_val
            mean_error = 0.0
