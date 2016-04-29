from __future__ import division

from copy import deepcopy

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
import time

import yaml

from edge.ei_helper import EI_LayerHelper, get_distance_mask

from edge.test.data import create_sample_data
from edge.networks import Basic_Network, Deep_Recurrent_Network


class MultivariateRNNTrainer(object):

    def __init__(self, hparams):
        self.hparams = hparams

        self.train_graph = None
        self.run_graph = None

        self.net = None
        self.optimizer = None
        self.train_vars = None
        self.run_vars = None

        self.epoch_errs = None
        self.epoch_test_errs = None
        self.test_preds = None

        self.trained_params = None

        self.build()

    def build(self):

        # build the graph that will execute for each batch
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():

            # construct the RNN
            self.net = Deep_Recurrent_Network(self.hparams)

            with tf.name_scope('training'):
                print("Building training op...")
                self.train_vars = self.create_batch_train_op()

            with tf.name_scope('running'):
                print("Building run op...")
                self.run_vars = self.create_run_op()

            # The optimizer
            with tf.name_scope('optimizer'):

                print("Building optimization ops...")

                t = tf.Variable(0, name= 't', trainable=False) # the step variable
                eta = None

                if self.hparams['opt_algorithm'] == 'adam':
                    # eta = tf.train.exponential_decay(self.hparams['eta0'], t, 2000, 0.5, staircase=True)
                    eta = tf.Variable(self.hparams['eta0'])
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=eta)

                elif self.hparams['opt_algorithm'] == 'annealed_sgd':
                    eta = tf.train.exponential_decay(self.hparams['eta0'], t, 5000, 0.1, staircase=True)
                    self.optimizer = tf.train.GradientDescentOptimizer(eta)

                grads, params = zip(*self.optimizer.compute_gradients(self.train_vars['batch_err']))
                # grads, _ = tf.clip_by_global_norm(grads, hparams['grad_clip_norm'])
                apply_grads = self.optimizer.apply_gradients(zip(grads, params), global_step=t)

            self.train_vars['eta'] = eta
            self.train_vars['grads'] = grads
            self.train_vars['params'] = params
            self.train_vars['apply_grads'] = apply_grads

    def create_batch_train_op(self):

        batch_size = self.hparams['batch_size']

        # the input placeholder, contains the entire multivariate input time series for a batch
        U = tf.placeholder(tf.float32, shape=[batch_size, self.hparams['t_mem'], self.hparams['n_in']])

        # the output placeholder, contains the desired multivariate output time series for a batch
        Y = tf.placeholder(tf.float32, shape=[batch_size, self.hparams['t_mem'], self.hparams['n_out']])

        # create placeholders for the initial state, one placeholder for each layer
        state_placeholders = list()
        for layer in self.net.layers:
            state_placeholders.append(tf.placeholder(tf.float32, shape=[batch_size, layer.n_unit]))

        next_state = state_placeholders
        weight_cost = self.net.weight_cost()

        # The forward propagation graph
        batch_err_list = list()
        for t in range(self.hparams['t_mem']):
            # construct the input vector at time t by slicing the input placeholder U, do the same for Y
            u = tf.reshape(tf.slice(U, [0, t, 0], [batch_size, 1, self.hparams['n_in']]), [batch_size, self.hparams['n_in']])
            y = tf.reshape(tf.slice(Y, [0, t, 0], [batch_size, 1, self.hparams['n_out']]), [batch_size, self.hparams['n_out']])

            # create an op to move the network state forward one time step, given the input
            # vector u and previous hidden state h. do this for all batches in parallel
            next_state, yhat = self.net.step(next_state, u)

            # compute the activity cost at time t across all batches
            activity_cost = self.net.activity_cost(next_state)

            # compute the mean squared error at time t, across batches
            mse = tf.reduce_mean(tf.reshape(tf.square(y - yhat), [-1]))

            # compute the total cost at time t, across batches, and append it to the list of costs
            total_cost = mse + weight_cost + activity_cost
            batch_err_list.append(tf.expand_dims(total_cost, 0))

        # compute the overall training error, the mean of the mean square error at each time point
        batch_err_list = tf.concat(0, batch_err_list)
        batch_err = tf.reduce_mean(batch_err_list)

        return {'U':U, 'Y':Y, 'state':state_placeholders, 'batch_err':batch_err, 'next_state':next_state}

    def create_run_op(self):
        # the input placeholder, contains the entire multivariate input time series for a batch
        U = tf.placeholder(tf.float32, shape=[self.hparams['t_mem_run'], self.hparams['n_in']])

        # create placeholders for the initial state, one placeholder for each layer
        state_placeholders = list()
        for layer in self.net.layers:
            state_placeholders.append(tf.placeholder(tf.float32, shape=[1, layer.n_unit]))

        next_state = state_placeholders

        # list to hold network outputs
        net_preds = list()
        for t in range(self.hparams['t_mem_run']):
            # construct the input vector at time t by slicing the input placeholder U, do the same for Y
            u = tf.slice(U, [t, 0], [1, self.hparams['n_in']])

            # create an op to move the network state forward one time step, given the input
            # vector u and previous hidden state h
            next_state, yhat = self.net.step(next_state, u)
            net_preds.append(yhat)

        return {'net_preds':net_preds, 'state':state_placeholders, 'U':U, 'next_state':next_state}

    def train(self, Utrain, Ytrain, Utest, Ytest, test_check_interval=5, plot_layers_during_training=False):

        n_train_steps = self.hparams['n_train_steps']
        
        n_batches,n_segs_per_batch,t_mem,n_in = Utrain.shape
        n_batches2,n_segs_per_batch2,t_mem2,n_out = Ytrain.shape
        assert n_batches == n_batches2
        assert n_segs_per_batch == n_segs_per_batch2
        assert t_mem == t_mem2

        # create a random initial state to start with
        h0_orig = self.net.initial_state(n_batches)
        h0 = h0_orig

        test_check = False
        if Utest is not None:
            assert Ytest is not None, "Must specify both Utest and Ytest, or neither!"
            test_check = True

        with tf.Session(graph=self.train_graph) as session:
            print('Starting training session...')

            tf.initialize_all_variables().run()

            # for each training iteration
            converged = False
            epoch_errs = list()
            epoch_test_errs = list()
            lowest_mean_err = np.inf

            for step in range(n_train_steps):

                if converged:
                    break

                # print("step=%d, len(all_variables())=%d" % (step, len(tf.all_variables())))

                stime = time.time()
                eta_val = None
                seg_errs = list()
                for seg in range(n_segs_per_batch):
                    # put the input and output matrices in the feed dictionary for this minibatch
                    Uk = Utrain[:, seg, :, :]
                    Yk = Ytrain[:, seg, :, :]
                    feed_dict = {self.train_vars['U']:Uk, self.train_vars['Y']:Yk}
                    for k,state_placeholder in enumerate(self.train_vars['state']):
                        feed_dict[state_placeholder] = h0[k]

                    to_compute = [self.train_vars['batch_err'], self.train_vars['eta'], self.train_vars['apply_grads']]
                    to_compute.extend(self.train_vars['next_state'])
                    compute_vals = session.run(to_compute, feed_dict=feed_dict)
                    train_error_val, eta_val = compute_vals[:2]
                    hnext_val = compute_vals[3:]

                    # get the last hidden state value to use on the next minibatch
                    h0 = hnext_val
                    seg_errs.append(train_error_val)

                seg_errs = np.array(seg_errs)
                epoch_errs.append(seg_errs)

                mean_seg_err = seg_errs.mean()
                if mean_seg_err < lowest_mean_err:
                    lowest_mean_err = mean_seg_err
                elif mean_seg_err > 1.03*lowest_mean_err:
                    print("Stopping optimization after this iteration because training error has increased too much.")
                    converged = True

                test_err = np.nan
                if test_check and ((step % test_check_interval == 0) or (step == n_train_steps-1)):
                    # average initial states across batches to get an initial state for the test set
                    h0_mean = list()
                    for h in h0:
                        nb,nu = h.shape
                        h0_mean.append(h.mean(axis=0).reshape([1, nu]))

                    Yhat = self.run_network(Utest, h0_mean, session)
                    test_err = np.mean((Yhat - Ytest)**2)
                    epoch_test_errs.append((step, test_err))
                    self.test_preds = Yhat

                etime = time.time() - stime
                print('iter=%d, eta=%0.6f, train_err=%0.6f +/- %0.6f, test_err=%0.6f, time=%0.3fs' % \
                      (step, eta_val, mean_seg_err, seg_errs.std(ddof=1), test_err, etime))

                if plot_layers_during_training:
                    for k,layer in enumerate(self.net.layers):
                        layer_params = layer.get_saveable_params(session)
                        layer.plot(layer_params)
                        plt.suptitle('Iter %d: Layer %d, type=%s' % (step, k, layer.hparams['rnn_type']))
                    plt.show()

            self.epoch_errs = np.array(epoch_errs)
            self.epoch_test_errs = np.array(epoch_test_errs)

            self.trained_params = dict()
            for k,layer in enumerate(self.net.layers):
                ldict = layer.get_saveable_params(session)
                self.trained_params['layer%d' % k] = ldict

    def run_network(self, U, h0, session):
        n_segs,t_mem_run,n_in = U.shape
        Yhat = np.zeros([t_mem_run*n_segs, self.hparams['n_out']])
        h = h0

        to_compute = list()
        npreds = len(self.run_vars['net_preds'])
        to_compute.extend(self.run_vars['net_preds'])
        to_compute.extend(self.run_vars['next_state'])

        for k in range(n_segs):
            si = k*t_mem_run
            ei = si + t_mem_run

            fdict = {self.run_vars['U']:U[k, :, :]}
            for k,state_placeholder in enumerate(self.run_vars['state']):
                fdict[state_placeholder] = h[k]
            compute_vals = session.run(to_compute, feed_dict=fdict)
            ypred = np.array(compute_vals[:npreds]).squeeze()
            h = compute_vals[npreds:]
            Yhat[si:ei, :] = ypred

        return Yhat.reshape([n_segs, t_mem_run, self.hparams['n_out']])

    def plot(self, Utest, Ytest, text_only=False):
        n_train_steps = self.hparams['n_train_steps']

        if not text_only:
            # make a figure that shows the training and validation error over epochs
            plt.figure()
            gs = plt.GridSpec(100, 100)

            train_err_mean = self.epoch_errs.mean(axis=1)
            train_err_std = self.epoch_errs.std(axis=1, ddof=1)
            ax = plt.subplot(gs[:45, :30])
            plt.errorbar(range(1, n_train_steps), train_err_mean[1:], yerr=train_err_std[1:],
                         linewidth=4.0, c='r', alpha=0.7, elinewidth=2.0, ecolor='k')
            plt.axis('tight')
            plt.xlabel('Epoch')
            plt.ylabel('Training Error')
            plt.title('Training Error')

            ax = plt.subplot(gs[50:, :30])
            test_x = self.epoch_test_errs[:, 0]
            test_err = self.epoch_test_errs[:, 1]
            plt.plot(test_x, test_err, linewidth=4.0, c='b', alpha=0.7)
            plt.axis('tight')
            plt.xlabel('Epoch')
            plt.ylabel('Test Error')
            plt.title('Test Error')

        n_segs_test,t_test_run,n_out = self.test_preds.shape
        ns,nt,nf = Ytest.shape
        assert nf == n_out
        nrows_per_plot = int(100. / nf)
        row_padding = 5

        for k in range(nf):
            # compute the correlation coefficient for each output prediction
            yt = Ytest[:, :, k].reshape([n_segs_test*t_test_run])
            yhatt = self.test_preds[:, :, k].reshape([n_segs_test*t_test_run])
            ycc = np.corrcoef(yt, yhatt)[0, 1]

            if text_only:
                print("Output %d: cc=%0.2f" % (k, ycc))

            else:
                # make a plot of the validation output series and the prediction
                gs_i = k*nrows_per_plot
                gs_e = ((k+1)*nrows_per_plot)-row_padding
                ax = plt.subplot(gs[gs_i:gs_e, 35:])

                plt.plot(yt, 'k-', linewidth=4.0, alpha=0.7)
                plt.plot(yhatt, 'r-', linewidth=4.0, alpha=0.7)
                plt.axis('tight')
                # plt.xlabel('Time')
                plt.ylabel('Y(t)')
                plt.legend(['Real', 'Prediction'])
                plt.title('cc=%0.2f' % ycc)

        if not text_only:
            n_layers = len(self.trained_params)
            for k in range(n_layers):
                lkey = 'layer%d' % k
                ldict = self.trained_params[lkey]
                self.net.layers[k].plot(ldict)

    def save(self, output_file):

        hf = h5py.File(output_file, 'w')

        hf.attrs['n_layers'] = len(self.net.layers)
        for k,v in self.hparams.items():
            if np.isscalar(v) or isinstance(v, str):
                hf.attrs[k] = v

        hf['epoch_test_errs'] = self.epoch_test_errs
        hf['epoch_errs'] = self.epoch_errs
        hf['test_preds'] = self.test_preds

        for lkey,ldict in self.trained_params.items():
            lgrp = hf.create_group(lkey)
            for pname,pval in ldict.items():
                lgrp[pname] = pval

        hf.close()


def read_config(config_file, n_in, n_out):

    with open(config_file) as config_f:
        net_params = yaml.load(config_f.read())

    hparams = dict()
    hparams['n_in'] = n_in
    hparams['n_out'] = n_out
    for key,val in net_params.items():
        if key == 'layers':
            layers = list()

            for k,ldict in enumerate(val):
                if k == 0:
                    layer_n_in = n_in
                else:
                    layer_n_in = layers[k-1]['n_unit']

                ldict['n_in'] = layer_n_in
                if ldict['rnn_type'] == 'EI':
                    layers.append(EI_LayerHelper.parse_config(ldict, layer_n_in))
                else:
                    layers.append(ldict)

            hparams['layers'] = layers
        else:
            hparams[key] = val

    return hparams

if __name__ == '__main__':

    np.random.seed(123456)

    n_in = 2
    n_hid_data = 4
    n_out = 3
    t_in = 10000

    # the "memory" of the network, how many time steps BPTT is run for
    t_mem = 20

    # set the time length per batch (the effective
    assert t_in % t_mem == 0
    n_samps = int(t_in / t_mem)

    # create some fake data for training
    Utrain, Xtrain, Ytrain, sample_params = create_sample_data(n_in, n_hid_data, n_out, t_in, n_samps, segment_U=True)

    # segment the training data into parallel batches, each batch is a continuous temporal segment of data
    # broken down into segments of length t_mem
    nsegs = Utrain.shape[0]
    n_batches = 10
    Utrain = Utrain.reshape([n_batches, nsegs/n_batches, t_mem, n_in])
    Ytrain = Ytrain.reshape([n_batches, nsegs/n_batches, t_mem, n_out])

    print("Utrain.shape=" + str(Utrain.shape))
    print("Ytrain.shape=" + str(Ytrain.shape))

    # create some fake data for validation
    t_mem_run = 50
    t_test_total = 1000
    n_test_segs = int(t_test_total / t_mem_run)

    Utest,Xtest,Ytest,sample_params2 = create_sample_data(n_in, n_hid_data, n_out, t_test_total, n_test_segs, segment_U=True,
                                                          Win=sample_params['Win'],
                                                          W=sample_params['W'],
                                                          b=sample_params['b'],
                                                          Wout=sample_params['Wout'],
                                                          bout=sample_params['bout'])

    print("Utest.shape=" + str(Utest.shape))
    print("Ytest.shape=" + str(Ytest.shape))

    hparams = read_config('param/deep_srnn.yaml', n_in, n_out)

    print('')
    print('------ Network Params ------')
    for k,v in hparams.items():
        if k != 'layers':
            print('%s=%s' % (k, str(v)))

    for k,ldict in enumerate(hparams['layers']):
        print("Layer %d: type=%s, n_in=%d, n_unit=%d, activation=%s" %
              (k, ldict['rnn_type'], ldict['n_in'], ldict['n_unit'], ldict['activation']))

    print("Building network...")
    rnn_trainer = MultivariateRNNTrainer(hparams)
    print("Training network...")
    rnn_trainer.train(Utrain, Ytrain, Utest, Ytest, plot_layers_during_training=False)
    rnn_trainer.plot(Utest, Ytest, text_only=False)
    plt.show()

    rnn_trainer.save('/tmp/rnn.h5')

    # print("Plotting network...")
