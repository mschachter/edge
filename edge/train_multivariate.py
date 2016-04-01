from __future__ import division

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from edge.test.data import create_sample_data
from edge.networks import Basic_Network


class MultivariateRNNTrainer(object):

    def __init__(self, hparams):
        self.hparams = hparams

        self.graph = None
        self.net = None
        self.optimizer = None
        self.vars = None

        self.train_errs_per_epoch = None
        self.test_errs_per_epoch = None
        self.test_preds = None

        self.build()

    def build(self):

        # build the graph that will execute for each batch
        self.graph = tf.Graph()
        with self.graph.as_default():

            # construct the RNN
            self.net = Basic_Network(n_in, hparams, n_output=n_out)

            with tf.name_scope('training'):

                # the input placeholder, contains the entire multivariate input time series for a batch
                U = tf.placeholder(tf.float32, shape=[t_mem, n_in])

                # the output placeholder, contains the desired multivariate output time series for a batch
                Y = tf.placeholder(tf.float32, shape=[t_mem, n_out])

                # the initial state placeholder, contains the initial state used at the beginning of a batch
                h = tf.placeholder(tf.float32, shape=[1, n_hid])

                hnext = h

                lambda2_val = hparams['lambda2']
                l2_cost = self.net.l2_W(lambda2_val) + self.net.l2_R(lambda2_val) + self.net.l2_Wout(lambda2_val)

                # The forward propagation graph
                net_err_list = list()
                net_preds = list()
                for t in range(t_mem):

                    # construct the input vector at time t by slicing the input placeholder U, do the same for Y
                    u = tf.slice(U, [t, 0], [1, n_in])
                    y = tf.slice(Y, [t, 0], [1, n_out])

                    # create an op to move the network state forward one time step, given the input
                    # vector u and previous hidden state h
                    (hnext,), yhat = self.net.step((hnext,), u)

                    # save the prediction op for this time step
                    net_preds.append(yhat)

                    # compute the cost at time t
                    mse = tf.reduce_mean(tf.squeeze(tf.square(y - yhat)))
                    err = mse + l2_cost
                    net_err_list.append(tf.expand_dims(err, 0))

                # compute the overall training error, the mean of the mean square error at each time point
                net_err_list = tf.concat(0, net_err_list)
                net_err = tf.reduce_mean(net_err_list)

            # The optimizer
            with tf.name_scope('optimizer'):

                t = tf.Variable(0, name= 't', trainable=False) # the step variable
                eta = None

                if hparams['opt_algorithm'] == 'adam':
                    eta = tf.train.exponential_decay(.008, t, 2000, 0.5, staircase=True)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=eta)

                elif hparams['opt_algorithm'] == 'annealed_sgd':
                    eta = tf.train.exponential_decay(5e-2, t, 5000, 0.1, staircase=True)
                    self.optimizer = tf.train.GradientDescentOptimizer(eta)

                grads, params = zip(*self.optimizer.compute_gradients(net_err))

                # grads, _ = tf.clip_by_global_norm(grads, hparams['grad_clip_norm'])

                apply_grads = self.optimizer.apply_gradients(zip(grads, params), global_step=t)

            self.vars = {'U':U, 'Y':Y, 'h':h,  't':t,  'hnext':hnext, 'l2_cost':l2_cost,
                         'net_err':net_err, 'net_preds':net_preds,
                         'eta':eta, 'grads':grads, 'params':params, 'apply_grads':apply_grads}

    def train(self, Utrain, Ytrain, Utest, Ytest):

        n_train_steps = self.hparams['n_train_steps']

        # create a random initial state to start with
        h0_orig = np.random.randn(n_hid) * 1e-3
        h0_orig = h0_orig.reshape([1, n_hid])
        h0 = h0_orig

        train_errs_per_epoch = list()
        test_errs_per_epoch = list()
        test_preds = None
        with tf.Session(graph=self.graph) as session:

            tf.initialize_all_variables().run()

            # for each training iteration
            for step in range(n_train_steps):

                h0_start = deepcopy(h0)

                train_errs_per_samp = list()
                test_errs_per_samp = list()

                # train on each minibatch
                for k in range(n_samps):
                    # put the input and output matrices in the feed dictionary for this minibatch
                    Uk = Utrain[k, :, :]
                    Yk = Ytrain[k, :, :]
                    feed_dict = {self.vars['U']:Uk, self.vars['Y']:Yk, self.vars['h']:h0}

                    # run the session to train the model for this minibatch
                    to_compute = [self.vars[vname] for vname in ['net_err', 'eta', 'hnext', 'apply_grads']]
                    train_error_val, eta_val, hnext_val = session.run(to_compute, feed_dict=feed_dict)[:3]

                    # get the last hidden state value to use on the next minibatch
                    h0 = hnext_val
                    train_errs_per_samp.append(train_error_val)

                # predict on the test set
                h0 = h0_start
                Yhat = list()
                for k in range(n_samps_test):
                    Uk = Utest[k, :, :]
                    Yk = Ytest[k, :, :]
                    feed_dict = {self.vars['U']:Uk, self.vars['Y']:Yk, self.vars['h']:h0}

                    to_compute = [self.vars[vname] for vname in ['net_err', 'hnext']]
                    to_compute.extend(self.vars['net_preds'])
                    compute_outputs = session.run(to_compute, feed_dict=feed_dict)
                    test_error_val, hnext_val = compute_outputs[:2]
                    test_preds_val = np.array(compute_outputs[2:])
                    test_errs_per_samp.append(test_error_val)
                    h0 = hnext_val
                    Yhat.append(test_preds_val)

                # overwrite the old value of test_preds, so that it's equal to the prediction
                # for the last optimization iteration
                test_preds = np.array(Yhat)
                
                train_errs_per_samp = np.array(train_errs_per_samp)
                test_errs_per_samp = np.array(test_errs_per_samp)
                
                print('iter=%d, eta=%0.6f, train_err=%0.6f +/- %0.3f, test_err=%0.6f +/- %0.3f' %
                  (step, eta_val,
                   train_errs_per_samp.mean(), train_errs_per_samp.std(ddof=1),
                   test_errs_per_samp.mean(), test_errs_per_samp.std(ddof=1)))

                train_errs_per_epoch.append(train_errs_per_samp)
                test_errs_per_epoch.append(test_errs_per_samp)

            self.train_errs_per_epoch = np.array(train_errs_per_epoch)
            self.test_errs_per_epoch = np.array(test_errs_per_epoch)

            self.test_preds = test_preds.squeeze()

    def plot_training(self, Utest, Ytest):

        n_train_steps = self.hparams['n_train_steps']

        plt.figure()
        gs = plt.GridSpec(100, 100)

        ax = plt.subplot(gs[:45, :30])
        plt.errorbar(range(n_train_steps), self.train_errs_per_epoch.mean(axis=1), yerr=self.train_errs_per_epoch.std(axis=1, ddof=1),
                     linewidth=4.0, c='r', alpha=0.7, elinewidth=2.0, ecolor='k')
        plt.axis('tight')
        plt.xlabel('Epoch')
        plt.ylabel('Training Error')
        plt.title('Training Error')

        ax = plt.subplot(gs[50:, :30])
        plt.errorbar(range(n_train_steps), self.test_errs_per_epoch.mean(axis=1), yerr=self.test_errs_per_epoch.std(axis=1, ddof=1),
                     linewidth=4.0, c='b', alpha=0.7, elinewidth=2.0, ecolor='k')
        plt.axis('tight')
        plt.xlabel('Epoch')
        plt.ylabel('Test Error')
        plt.title('Test Error')

        Y_t = Ytest.reshape([Ytest.shape[0]*Ytest.shape[1], Ytest.shape[2]])
        Yhat_t = self.test_preds.reshape([self.test_preds.shape[0]*self.test_preds.shape[1], self.test_preds.shape[2]])

        nt,nf = Y_t.shape
        nrows_per_plot = int(100. / nf)
        row_padding = 5

        for k in range(nf):
            gs_i = k*nrows_per_plot
            gs_e = ((k+1)*nrows_per_plot)-row_padding
            ax = plt.subplot(gs[gs_i:gs_e, 35:])

            yt = Y_t[:, k]
            yhatt = Yhat_t[:, k]
            ycc = np.corrcoef(yt, yhatt)[0, 1]

            plt.plot(yt, 'k-', linewidth=4.0, alpha=0.7)
            plt.plot(yhatt, 'r-', linewidth=4.0, alpha=0.7)
            plt.axis('tight')
            # plt.xlabel('Time')
            plt.ylabel('Y(t)')
            plt.legend(['Real', 'Prediction'])
            plt.title('cc=%0.2f' % ycc)

        plt.show()


if __name__ == '__main__':

    np.random.seed(123456)

    n_in = 2
    n_hid_data = 10
    n_hid = 25
    n_out = 16
    t_in = 2000

    # the "memory" of the network, how many time steps BPTT is run for
    t_mem = 20

    # set the time length per batch (the effective
    assert t_in % t_mem == 0
    n_samps = int(t_in / t_mem)

    # create some fake data for training
    Utrain, Xtrain, Ytrain, sample_params = create_sample_data(n_in, n_hid_data, n_out, t_in, n_samps, segment_U=True)

    """
    plt.figure()

    ax = plt.subplot(1, 2, 1)
    plt.hist(Utrain.ravel(), bins=25, color='r', alpha=0.7)
    plt.axis('tight')
    plt.title('Distribution of Inputs')

    ax = plt.subplot(1, 2, 2)
    plt.hist(Ytrain.ravel(), bins=25, color='b', alpha=0.7)
    plt.axis('tight')
    plt.title('Distribution of Outputs')

    plt.show()
    """

    # create some fake data for validation
    t_in_test = 100
    n_samps_test = int(t_in_test / t_mem)
    Utest,Xtest,Ytest,sample_params2 = create_sample_data(n_in, n_hid_data, n_out, t_in_test, n_samps_test, segment_U=True,
                                                          Win=sample_params['Win'],
                                                          W=sample_params['W'],
                                                          b=sample_params['b'],
                                                          Wout=sample_params['Wout'],
                                                          bout=sample_params['bout'])

    hparams = {'rnn_type':'SRNN', 'opt_algorithm':'annealed_sgd',
               'n_train_steps':75, 'n_unit':n_hid, 'dropout':{'R':0.0, 'W':0.0},
               'lambda2':1e-1}

    rnn_trainer = MultivariateRNNTrainer(hparams)
    rnn_trainer.train(Utrain, Ytrain, Utest, Ytest)
    rnn_trainer.plot_training(Utest, Ytest)
