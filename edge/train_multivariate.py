from __future__ import division

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import time

from edge.test.data import create_sample_data
from edge.networks import Basic_Network


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
            self.net = Basic_Network(self.hparams['n_in'], self.hparams, n_output=self.hparams['n_out'])

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
            self.train_vars['R'] = self.net.rnn_layer.R

            if 'sign_constrain_R' in self.hparams:
                self.train_vars['sign_constrain_R'] = self.net.sign_constraint_R(hparams['sign_constrain_R'])

    def create_batch_train_op(self):

        batch_size = self.hparams['batch_size']

        # the input placeholder, contains the entire multivariate input time series for a batch
        U = tf.placeholder(tf.float32, shape=[batch_size, self.hparams['t_mem'], self.hparams['n_in']])

        # the output placeholder, contains the desired multivariate output time series for a batch
        Y = tf.placeholder(tf.float32, shape=[batch_size, self.hparams['t_mem'], self.hparams['n_out']])

        # the initial state placeholder, contains the initial state used at the beginning of a batch,
        # for each batch
        h = tf.placeholder(tf.float32, shape=[batch_size, self.hparams['n_unit']])
        hnext = h
        lambda2_val = self.hparams['lambda2']
        l2_cost = self.net.l2_W(lambda2_val) + self.net.l2_R(lambda2_val) + self.net.l2_Wout(lambda2_val)

        # The forward propagation graph
        batch_err_list = list()
        # net_preds = list()
        for t in range(self.hparams['t_mem']):
            # construct the input vector at time t by slicing the input placeholder U, do the same for Y
            u = tf.reshape(tf.slice(U, [0, t, 0], [batch_size, 1, self.hparams['n_in']]), [batch_size, self.hparams['n_in']])
            y = tf.reshape(tf.slice(Y, [0, t, 0], [batch_size, 1, self.hparams['n_out']]), [batch_size, self.hparams['n_out']])

            # create an op to move the network state forward one time step, given the input
            # vector u and previous hidden state h. do this for all batches in parallel
            (hnext,), yhat = self.net.step((hnext,), u)

            # save the prediction op for this time step
            # net_preds.append(yhat)

            # compute the cost at time t across all batches
            mse = tf.reduce_mean(tf.reshape(tf.square(y - yhat), [-1]))
            err = mse + l2_cost
            batch_err_list.append(tf.expand_dims(err, 0))

        # compute the overall training error, the mean of the mean square error at each time point
        batch_err_list = tf.concat(0, batch_err_list)
        batch_err = tf.reduce_mean(batch_err_list)

        return {'U':U, 'Y':Y, 'batch_err':batch_err, 'h':h, 'hnext':hnext}

    def create_run_op(self):
        # the input placeholder, contains the entire multivariate input time series for a batch
        U = tf.placeholder(tf.float32, shape=[self.hparams['t_mem_run'], self.hparams['n_in']])

        # the initial state placeholder, contains the initial state used at the beginning of a batch
        h = tf.placeholder(tf.float32, shape=[1, self.hparams['n_unit']])
        hnext = h

        # list to hold network outputs
        net_preds = list()
        for t in range(self.hparams['t_mem_run']):
            # construct the input vector at time t by slicing the input placeholder U, do the same for Y
            u = tf.slice(U, [t, 0], [1, self.hparams['n_in']])

            # create an op to move the network state forward one time step, given the input
            # vector u and previous hidden state h
            (hnext,), yhat = self.net.step((hnext,), u)
            net_preds.append(yhat)

        return {'net_preds':net_preds, 'h':h, 'U':U, 'hnext':hnext}

    def train(self, Utrain, Ytrain, Utest=None, Ytest=None, test_check_interval=5):

        n_train_steps = self.hparams['n_train_steps']
        
        n_batches,n_segs_per_batch,t_mem,n_in = Utrain.shape
        n_batches2,n_segs_per_batch2,t_mem2,n_out = Ytrain.shape
        assert n_batches == n_batches2
        assert n_segs_per_batch == n_segs_per_batch2
        assert t_mem == t_mem2

        # create a random initial state to start with
        h0_orig = np.random.randn(n_batches, self.hparams['n_unit']) * 1e-3
        h0 = h0_orig

        test_check = False
        if Utest is not None:
            assert Ytest is not None, "Must specify both Utest and Ytest, or neither!"
            test_check = True

        with tf.Session(graph=self.train_graph) as session:
            print('Starting training session...')

            tf.initialize_all_variables().run()

            # for each training iteration
            epoch_errs = list()
            epoch_test_errs = list()
            for step in range(n_train_steps):

                # print("step=%d, len(all_variables())=%d" % (step, len(tf.all_variables())))

                stime = time.time()
                eta_val = None
                seg_errs = list()
                for seg in range(n_segs_per_batch):

                    # put the input and output matrices in the feed dictionary for this minibatch
                    Uk = Utrain[:, seg, :, :]
                    Yk = Ytrain[:, seg, :, :]
                    feed_dict = {self.train_vars['U']:Uk, self.train_vars['Y']:Yk, self.train_vars['h']:h0}

                    # run the session to train the model for this minibatch
                    to_compute = [self.train_vars[vname] for vname in ['batch_err', 'eta', 'hnext', 'apply_grads']]
                    if 'sign_constrain_R' in self.train_vars:
                        to_compute.append(self.train_vars['sign_constrain_R'])

                    train_error_val, eta_val, hnext_val = session.run(to_compute, feed_dict=feed_dict)[:3]

                    # get the last hidden state value to use on the next minibatch
                    h0 = hnext_val
                    seg_errs.append(train_error_val)

                seg_errs = np.array(seg_errs)
                epoch_errs.append(seg_errs)

                test_err = np.nan
                if test_check and ((step % test_check_interval == 0) or (step == n_train_steps-1)):
                    # average initial states across batches to get an initial state for the test set
                    h0_mean = h0.mean(axis=0).reshape([1, self.hparams['n_unit']])
                    Yhat = self.run_network(Utest, h0_mean, session)
                    test_err = np.mean((Yhat - Ytest)**2)
                    epoch_test_errs.append((step, test_err))
                    self.test_preds = Yhat

                etime = time.time() - stime
                print('iter=%d, eta=%0.6f, train_err=%0.6f +/- %0.6f, test_err=%0.6f, time=%0.3fs' % \
                      (step, eta_val, seg_errs.mean(), seg_errs.std(ddof=1), test_err, etime))

            self.epoch_errs = np.array(epoch_errs)
            self.epoch_test_errs = np.array(epoch_test_errs)

            self.trained_params = {'R':session.run(self.train_vars['R'])}

    def run_network(self, U, h0, session):
        n_segs,t_mem_run,n_in = U.shape
        Yhat = np.zeros([t_mem_run*n_segs, self.hparams['n_out']])
        h = h0

        to_compute = list()
        to_compute.extend(self.run_vars['net_preds'])
        to_compute.append(self.run_vars['hnext'])

        for k in range(n_segs):
            si = k*t_mem_run
            ei = si + t_mem_run
            compute_vals = session.run(to_compute,
                                       feed_dict={self.run_vars['U']:U[k, :, :], self.run_vars['h']:h})
            ypred = np.array(compute_vals[:-1]).squeeze()
            h = np.array(compute_vals[-1])
            Yhat[si:ei, :] = ypred

        return Yhat.reshape([n_segs, t_mem_run, self.hparams['n_out']])

    def plot(self, Utest, Ytest, text_only=False):
        n_train_steps = self.hparams['n_train_steps']

        if not text_only:
            plt.figure()
            gs = plt.GridSpec(100, 100)

            ax = plt.subplot(gs[:45, :30])
            plt.errorbar(range(n_train_steps), self.epoch_errs.mean(axis=1), yerr=self.epoch_errs.std(axis=1, ddof=1),
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
            yt = Ytest[:, :, k].reshape([n_segs_test*t_test_run])
            yhatt = self.test_preds[:, :, k].reshape([n_segs_test*t_test_run])
            ycc = np.corrcoef(yt, yhatt)[0, 1]

            if text_only:
                print("Output %d: cc=%0.2f" % (k, ycc))

            else:
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
            plt.figure()
            R = self.trained_params['R']
            absmax = np.abs(R).max()
            plt.imshow(R, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
            plt.colorbar()
            plt.title('Recurrent Weight Matrix')

            plt.show()


if __name__ == '__main__':

    np.random.seed(123456)

    n_in = 2
    n_hid_data = 4
    n_hid = 20
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

    hparams = {'rnn_type':'SRNN', 'n_in':n_in, 'n_out':n_out, 'n_unit':n_hid, 'activation':'relu', 't_mem':t_mem,
               'opt_algorithm':'annealed_sgd', 'n_train_steps':25, 'batch_size':n_batches, 'eta0':5e-2,
               'dropout':{'R':0.0, 'W':0.0}, 'lambda2':1e-1,
               't_mem_run':Utest.shape[1],
               }

    sign_mat = np.ones([n_hid, n_hid])
    for k in range(n_hid):
        if k % 2 == 0:
            sign_mat[k, :] *= -1.

    hparams['sign_constrain_R'] = sign_mat

    print("Building network...")
    rnn_trainer = MultivariateRNNTrainer(hparams)
    print("Training network...")
    rnn_trainer.train(Utrain, Ytrain, Utest, Ytest)
    print("Plotting network...")
    rnn_trainer.plot(Utest, Ytest, text_only=False)
