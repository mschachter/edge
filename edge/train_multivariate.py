from __future__ import division

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
                    eta = tf.train.exponential_decay(self.hparams['eta0'], t, 2000, 0.5, staircase=True)
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

            if 'sign_constrain_R' in hparams:
                self.train_vars['sign_constrain_R'] = self.net.sign_constraint_R(hparams['sign_constrain_R'])

    def create_batch_train_op(self):

        # the input placeholder, contains the entire multivariate input time series for a batch
        U = tf.placeholder(tf.float32, shape=[self.hparams['batch_size'], self.hparams['t_mem'], self.hparams['n_in']])

        # the output placeholder, contains the desired multivariate output time series for a batch
        Y = tf.placeholder(tf.float32, shape=[self.hparams['batch_size'], self.hparams['t_mem'], self.hparams['n_out']])

        # the initial state placeholder, contains the initial state used at the beginning of a batch
        h = tf.placeholder(tf.float32, shape=[1, self.hparams['n_unit']])
        hnext = h
        lambda2_val = self.hparams['lambda2']
        l2_cost = self.net.l2_W(lambda2_val) + self.net.l2_R(lambda2_val) + self.net.l2_Wout(lambda2_val)

        # The forward propagation graph
        batch_err_list = list()
        # batch_net_preds = list()
        for b in range(self.hparams['batch_size']):

            net_err_list = list()
            # net_preds = list()
            for t in range(self.hparams['t_mem']):
                # construct the input vector at time t by slicing the input placeholder U, do the same for Y
                u = tf.reshape(tf.slice(U, [b, t, 0], [1, 1, self.hparams['n_in']]), [1, self.hparams['n_in']])
                y = tf.reshape(tf.slice(Y, [b, t, 0], [1, 1, self.hparams['n_out']]), [1, self.hparams['n_out']])

                # create an op to move the network state forward one time step, given the input
                # vector u and previous hidden state h
                (hnext,), yhat = self.net.step((hnext,), u)

                # save the prediction op for this time step
                # net_preds.append(yhat)

                # compute the cost at time t
                mse = tf.reduce_mean(tf.squeeze(tf.square(y - yhat)))
                err = mse + l2_cost
                net_err_list.append(tf.expand_dims(err, 0))

            # compute the overall training error, the mean of the mean square error at each time point
            net_err_list = tf.concat(0, net_err_list)
            net_err = tf.reduce_mean(net_err_list)
            batch_err_list.append(tf.expand_dims(net_err, 0))
            # batch_net_preds.append(net_preds)
        batch_err_list = tf.concat(0, batch_err_list)
        batch_err = tf.reduce_mean(batch_err_list)

        return {'U':U, 'Y':Y, 'batch_err':batch_err, 'h':h, 'hnext':hnext}

    def create_run_op(self):

        # the input placeholder, contains the entire multivariate input time series for a batch
        U = tf.placeholder(tf.float32, shape=[self.hparams['t_run'], self.hparams['n_in']])

        # the initial state placeholder, contains the initial state used at the beginning of a batch
        h = tf.placeholder(tf.float32, shape=[1, self.hparams['n_unit']])
        hnext = h

        # list to hold network outputs
        net_preds = list()

        for t in range(self.hparams['t_run']):
            # construct the input vector at time t by slicing the input placeholder U, do the same for Y
            u = tf.slice(U, [t, 0], [1, self.hparams['n_in']])

            # create an op to move the network state forward one time step, given the input
            # vector u and previous hidden state h
            (hnext,), yhat = self.net.step((hnext,), u)
            net_preds.append(yhat)

        return {'net_preds':net_preds, 'h':h, 'U':U, 'hnext':hnext}

    def train(self, Utrain, Ytrain, Utest=None, Ytest=None, test_check_interval=5):

        n_train_steps = self.hparams['n_train_steps']
        n_samps = Utrain.shape[0]
        assert Ytrain.shape[0] == n_samps

        n_samps_per_batch = self.hparams['batch_size']
        n_batches = int(n_samps / n_samps_per_batch)
        print('# of samples: %d' % n_samps)
        print('# of batches: %d' % n_batches)

        # create a random initial state to start with
        h0_orig = np.random.randn(self.hparams['n_unit']) * 1e-3
        h0_orig = h0_orig.reshape([1, self.hparams['n_unit']])
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
                batch_errs = list()
                for b in range(n_batches):
                    batch_i = b*n_samps_per_batch
                    batch_e = batch_i + n_samps_per_batch

                    # put the input and output matrices in the feed dictionary for this minibatch
                    Uk = Utrain[batch_i:batch_e, :, :]
                    Yk = Ytrain[batch_i:batch_e, :, :]
                    feed_dict = {self.train_vars['U']:Uk, self.train_vars['Y']:Yk, self.train_vars['h']:h0}

                    # run the session to train the model for this minibatch
                    to_compute = [self.train_vars[vname] for vname in ['batch_err', 'eta', 'hnext', 'apply_grads']]
                    if 'sign_constrain_R' in self.train_vars:
                        to_compute.append(self.train_vars['sign_constrain_R'])

                    train_error_val, eta_val, hnext_val = session.run(to_compute, feed_dict=feed_dict)[:3]

                    # get the last hidden state value to use on the next minibatch
                    h0 = hnext_val
                    batch_errs.append(train_error_val)

                batch_errs = np.array(batch_errs)
                epoch_errs.append(batch_errs)

                test_err = np.nan
                if test_check and ((step % test_check_interval == 0) or (step == n_train_steps-1)):
                    Yhat = session.run(self.run_vars['net_preds'], feed_dict={self.run_vars['U']:Utest, self.run_vars['h']:h0})
                    Yhat = np.array(Yhat).squeeze()
                    test_err = np.mean((Yhat - Ytest)**2)
                    epoch_test_errs.append((step, test_err))
                    self.test_preds = Yhat

                etime = time.time() - stime
                print('iter=%d, eta=%0.6f, train_err=%0.6f +/- %0.6f, test_err=%0.6f, time=%0.3fs' % \
                      (step, eta_val, batch_errs.mean(), batch_errs.std(ddof=1), test_err, etime))

            self.epoch_errs = np.array(epoch_errs)
            self.epoch_test_errs = np.array(epoch_test_errs)

            self.trained_params = {'R':session.run(self.train_vars['R'])}

    def plot(self, Utest, Ytest):
        n_train_steps = self.hparams['n_train_steps']

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

        Yhat_t = self.test_preds
        nt,nf = Ytest.shape
        nrows_per_plot = int(100. / nf)
        row_padding = 5

        for k in range(nf):
            gs_i = k*nrows_per_plot
            gs_e = ((k+1)*nrows_per_plot)-row_padding
            ax = plt.subplot(gs[gs_i:gs_e, 35:])

            yt = Ytest[:, k]
            yhatt = Yhat_t[:, k]
            ycc = np.corrcoef(yt, yhatt)[0, 1]

            plt.plot(yt, 'k-', linewidth=4.0, alpha=0.7)
            plt.plot(yhatt, 'r-', linewidth=4.0, alpha=0.7)
            plt.axis('tight')
            # plt.xlabel('Time')
            plt.ylabel('Y(t)')
            plt.legend(['Real', 'Prediction'])
            plt.title('cc=%0.2f' % ycc)

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
    t_in = 5000

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
    t_in_test = 1000
    n_samps_test = int(t_in_test / t_mem)
    Utest,Xtest,Ytest,sample_params2 = create_sample_data(n_in, n_hid_data, n_out, t_in_test, n_samps_test, segment_U=False,
                                                          Win=sample_params['Win'],
                                                          W=sample_params['W'],
                                                          b=sample_params['b'],
                                                          Wout=sample_params['Wout'],
                                                          bout=sample_params['bout'])

    hparams = {'rnn_type':'SRNN', 'opt_algorithm':'annealed_sgd', 'n_train_steps':25, 'batch_size':1,
               'n_in':n_in, 'n_out':n_out, 'n_unit':n_hid, 'activation':'relu',
               'dropout':{'R':0.0, 'W':0.0}, 'lambda2':1e-1, 't_mem':t_mem, 't_run':Utest.shape[0],
               'eta0':5e-2}

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
    rnn_trainer.plot(Utest, Ytest)
