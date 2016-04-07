from __future__ import division
import tensorflow as tf
import yaml
import time
from datetime import datetime
import sys, os
import pprint
import h5py
import numpy as np
import json

from input_generator import Input_Generator
import util.text_processing as text_proc
from networks import Prediction_Network
from sampler import Sampler

np.random.seed(0)
tf.set_random_seed(0)

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
    text_proc.file_to_datasets(data_path, valid_fraction = .01)
n_alphabet = len(alphabet)
hparams['n_alphabet'] = n_alphabet
hparams['alphabet'] = alphabet.tolist()


n_train = len(train_text)

n_batch = hparams['n_batch']
n_prop = hparams['n_prop']

train_input_generator = Input_Generator(train_text, alphabet, n_batch, n_prop)




graph = tf.Graph()
with graph.as_default():

    with tf.name_scope('network_parameters'):
        net = Prediction_Network(hparams)

    ## First we build the training graph

    # The network and it training state
    with tf.name_scope('input_data'):
        # The input nodes
        xs = [tf.placeholder(tf.float32, shape=[n_batch, n_alphabet]) for
            _ in xrange(n_prop + 1)]
        x_inputs = xs[:-1]
        x_labels = xs[1:]

        # The forward propagation graph
        errs = list()
        entropies = list()

    with tf.name_scope('bptt_graph'):


        with tf.name_scope('t_0'):
            train_state_store = net.get_new_state_store(n_batch)
            train_state = net.state_from_store(train_state_store)


        for i in range(len(x_inputs)):
            x_input = x_inputs[i]
            x_label = x_labels[i]

            with tf.name_scope('t_' + str(i+1)):
                x_pred = net.step(train_state, x_input)
                cross_entropy, entropy = net.evaluate_prediction(train_state, x_pred, x_label)

                if i > hparams['n_reject']:
                    errs.append(cross_entropy)

        # The update that allows state to carry across f-props
        with tf.name_scope('save_state'):
            store_train_state = net.store_state_op(train_state, train_state_store)
            reset_train_state = net.reset_state_op(train_state_store)


    with tf.name_scope('mean_error'):
        train_err = tf.reduce_mean(tf.concat(0, errs))
        train_summ = tf.scalar_summary('train error', train_err)



    # The optimizer
    with tf.name_scope('optimizer'):
        t = tf.Variable(0, name= 't', trainable=False) # the step variable

        if hparams['opt_algorithm'] == 'adam':
            eta = tf.train.exponential_decay(.008, t, 2000, 0.5, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=eta)
        elif hparams['opt_algorithm'] == 'annealed_sgd':
            eta = tf.train.exponential_decay(1.0, t, 5000, 0.1, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(eta)

        grads, params = zip(*optimizer.compute_gradients(train_err))
        grads, _ = tf.clip_by_global_norm(grads, hparams['grad_clip_norm'])
        apply_grads = optimizer.apply_gradients(zip(grads, params), global_step=t)

    with tf.name_scope('sampler') as scope:
        sampler = Sampler(net, alphabet)


    saver = tf.train.Saver()


n_train_steps = hparams['n_train_steps']

summary_freq = 100
mean_error = 0.0


train_error_hist = []
valid_error_hist = []

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('/tmp/new_imp', graph_def = session.graph_def)

    for step in range(n_train_steps):

        reset_rate = n_prop/hparams['train_state_reset_rate']
        if np.random.poisson(reset_rate) > 0:
            print 'State reset!'
            session.run([reset_train_state])

        # Set up input value -> input var mapping
        window = train_input_generator.next_window()
        feed_dict = dict()
        for i in range(n_prop + 1):
            feed_dict[xs[i]] = window[i]

        to_compute = [merged, train_err, eta, apply_grads, store_train_state]
        summ_str, error_val, eta_val = session.run(to_compute, feed_dict=feed_dict)[:3]
        writer.add_summary(summ_str, step)

        mean_error += error_val

        if step % summary_freq == 0 and step > 0:
            mean_error = mean_error/summary_freq

            train_error_hist.append((step, mean_error))

            print 'Average error at step', step, ':', mean_error, 'learning rate:', eta_val
            mean_error = 0.0

            if step % (summary_freq*10) == 0:

                test_xents, test_ents = sampler.test_prediction_error(session, valid_text)
                mean_valid_error = np.mean(test_xents)

                valid_error_hist.append((step, mean_valid_error))

                print 'Validation error:', mean_valid_error

                prime, sample_string = sampler.sample(session, bias = 10.0)
                print 'Sampling... ' + prime + '-->' + sample_string

    # save the model, params, and stats to the run dir
    model_file = os.path.join(run_path, 'model.ckpt')
    saver.save(session, model_file)

    train_short = train_text[:10000]
    train_xents, train_ents = sampler.test_prediction_error(session, train_short)
    test_xents, test_ents = sampler.test_prediction_error(session, test_text)

json_file = os.path.join(run_path, 'test_stats.json')
with open(json_file, 'w') as f:
    json.dump({'seq': alphabet[test_text[1:].tolist()].tolist(), 'cross_entropies': test_xents.tolist(),
        'entropies': test_ents.tolist()}, f)

json_file = os.path.join(run_path, 'train_stats.json')
with open(json_file, 'w') as f:
    json.dump({'seq': alphabet[train_short[1:].tolist()].tolist(), 'cross_entropies': train_xents.tolist(),
        'entropies': train_ents.tolist()}, f)




def write_dict_to_hdf5(filename, dictionary):
    with h5py.File(filename, 'w') as hdf5_file:
        for key in dictionary:
            hdf5_file.create_dataset(key, data=dictionary[key])

train_error_hist = np.array(zip(*train_error_hist))
valid_error_hist = np.array(zip(*valid_error_hist))
stats = {'train_error_hist': train_error_hist, 'valid_error_hist': valid_error_hist}
stats_file = os.path.join(run_path, 'stats.hdf5')
write_dict_to_hdf5(stats_file, stats)

params_file = os.path.join(run_path, 'params.yaml')
with open(params_file, 'w') as f:
    yaml.dump(hparams, f)
