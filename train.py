import tensorflow as tf
import yaml
import time
from datetime import datetime
import sys, os
import pprint
import h5py
import numpy as np

from input_generator import Input_Generator
import util.text_processing as text_proc
from networks import Basic_Network
from sampler import Sampler

t_run_start = time.time()

if len(sys.argv) > 1:
    param_file_path = sys.argv[1]
else:
    raise Exception('Specify a parameter file to run on!')

with open('param/defaults.yaml', 'r') as f:
    hparams = yaml.load(f)
with open(param_file_path, 'r') as f:
    net_params = yaml.load(f)
hparams.update(net_params)

print '-------------'
print 'Running on', param_file_path
pprint.pprint(hparams)
print '-------------'

if 'start_date' not in hparams:
    start_date = datetime.fromtimestamp(t_run_start)
    date_str = start_date.isoformat()
    hparams['start_date'] = date_str
else:
    date_str = hparams['start_date']

run_dir  = hparams['run_dir']
run_path = os.path.join(run_dir, hparams['run_name'] + '-' + date_str)
if not os.path.exists(run_path):
    os.makedirs(run_path)


data_path = os.path.join(hparams['data_dir'], hparams['data_file'])

train_text, valid_text, test_text, alphabet = \
    text_proc.file_to_datasets(data_path)
n_alphabet = len(alphabet)

n_train = len(train_text)
n_valid = len(valid_text)

n_batch = hparams['n_prop']
n_prop = hparams['n_batch']

train_input_generator = Input_Generator(train_text, alphabet, n_batch, n_prop)
valid_input_generator = Input_Generator(valid_text, alphabet, 1, 1)

n_unit = hparams['n_unit']




graph = tf.Graph()
with graph.as_default():

    ## First we build the training graph

    # The network and it training state
    with tf.name_scope('training') as scope:
        net = Basic_Network(n_alphabet, hparams)

        # The input nodes
        xs = [tf.placeholder(tf.float32, shape=[n_batch, n_alphabet]) for
            _ in xrange(n_prop + 1)]
        x_inputs = xs[:-1]
        x_labels = xs[1:]

        # The forward propagation graph
        errs = list()

        init_train_state = net.get_new_states(n_batch)
        train_state = init_train_state

        init_d_state = net.get_new_states(n_batch)
        d_state = init_d_state
        for (x_input, x_label) in zip(x_inputs, x_labels):
            train_state, logits = net.step(train_state, x_input, d_state)

            err = tf.nn.softmax_cross_entropy_with_logits(logits, x_label)
            errs.append(err)

            d_state = net.gradient(err, train_state)


        train_err = tf.reduce_mean(tf.concat(0, errs))
        train_summ = tf.scalar_summary('train error', train_err)

        # The update that allows state to carry across f-props
        store_train_state = net.store_state_op(train_state, init_train_state)
        store_d_state = net.store_state_op(d_state, init_d_state)

    # The optimizer
    with tf.name_scope('optimizer') as scope:
        t = tf.Variable(0, name= 't', trainable=False) # the step variable
        eta = tf.train.exponential_decay(10.0, t, 5000, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(eta)
        grads, params = zip(*optimizer.compute_gradients(train_err))
        grads, _ = tf.clip_by_global_norm(grads, hparams['grad_clip_norm'])
        apply_grads = optimizer.apply_gradients(zip(grads, params), global_step=t)

    ## Now we build the validation graph using the same parameters
    ## but a different state since we don't want batches when validating
    with tf.name_scope('validation') as scope:
        cur_valid_state = net.get_new_states(1)
        cur_valid_d_state = net.get_new_states(1)

        valid_input = tf.placeholder(tf.float32, shape=[1, n_alphabet])
        valid_label = tf.placeholder(tf.float32, shape=[1, n_alphabet])

        next_valid_state, logits = net.step(cur_valid_state, valid_input, cur_valid_d_state)

        valid_err = tf.nn.softmax_cross_entropy_with_logits(logits, valid_label)
        valid_summ = tf.scalar_summary('validation error', valid_err[0])

        next_valid_d_state = net.gradient(valid_err, next_valid_state)


        store_valid_state = net.store_state_op(next_valid_state, cur_valid_state)
        store_valid_d_state = net.store_state_op(next_valid_d_state, cur_valid_d_state)

        reset_valid_d_state = net.reset_state_op(cur_valid_d_state)
        reset_valid_state = net.reset_state_op(cur_valid_state)

    saver = tf.train.Saver()
    summaries = tf.merge_all_summaries()

    # sampler = Sampler(net, alphabet)

n_train_steps = hparams['n_train_steps']

summary_freq = 100
mean_error = 0.0


train_error_hist = []
valid_error_hist = []

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    # tf.train.SummaryWriter('/tmp/lstm', graph_def = session.graph_def)

    for step in range(n_train_steps):

        # Set up input value -> input var mapping
        window = train_input_generator.next_window()
        feed_dict = dict()
        for i in range(n_prop + 1):
            feed_dict[xs[i]] = window[i]

        to_compute = [train_err, eta, apply_grads, store_train_state]
        if net.uses_error:
            to_compute.append(store_d_state)
        error_val, eta_val = session.run(to_compute, feed_dict=feed_dict)[:2]

        mean_error += error_val

        if step % summary_freq == 0 and step > 0:
            mean_error = mean_error/summary_freq

            train_error_hist.append((step, mean_error))

            print 'Average error at step', step, ':', mean_error, 'learning rate:', eta_val
            mean_error = 0.0



            if step % (summary_freq*10) == 0:
                session.run(reset_valid_state)
                if net.uses_error:
                    session.run(reset_valid_d_state)

                mean_valid_error = 0
                for i in range(n_valid):
                    window = valid_input_generator.next_window()
                    feed_dict = {valid_input: window[0], valid_label:window[1]}
                    to_compute = [valid_err, store_valid_state]
                    if net.uses_error:
                        to_compute.append(store_valid_d_state)
                    valid_err_val = session.run(to_compute, feed_dict)[0]

                    mean_valid_error += valid_err_val[0]
                mean_valid_error /= n_valid

                valid_error_hist.append((step, mean_valid_error))

                print 'Validation error:', mean_valid_error

                # prime, sample_string = sampler.sample(session, bias = 2.0)
                # print 'Sampling... ' + prime + '-->' + sample_string

    # save the model, params, and stats to the run dir
    model_file = os.path.join(run_path, 'model')
    saver.save(session, model_file)



def write_dict_to_hdf5(file, dictionary):
    with h5py.File(file, 'w') as f:
        for key in dictionary:
            f.create_dataset(key, data=dictionary[key])

train_error_hist = np.array(zip(*train_error_hist))
valid_error_hist = np.array(zip(*valid_error_hist))
stats = {'train_error_hist': train_error_hist, 'valid_error_hist': valid_error_hist}
stats_file = os.path.join(run_path, 'stats')
write_dict_to_hdf5(stats_file, stats)

params_file = os.path.join(run_path, 'params')
with open(params_file, 'w') as f:
    yaml.dump(hparams, f)
