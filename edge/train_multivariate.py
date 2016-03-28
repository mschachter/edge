from __future__ import division

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from edge.test.data import create_sample_data
from edge.networks import Basic_Network

"""
U = tf.placeholder(tf.float32, shape=(10, 2))
Y = tf.placeholder(tf.float32, shape=(10, 2))
W = tf.constant(np.array([[-1., 2.], [0., 3]]).astype('float32'))

u2 = tf.slice(U, [2, 0], [1, 2])
y2 = tf.slice(Y, [2, 0], [1, 1])

u3 = tf.slice(U, [3, 0], [1, 2])
y3 = tf.slice(Y, [3, 0], [1, 1])

yhat2 = tf.matmul(u2, W)
yhat3 = tf.matmul(u3, W)

e2 = tf.reduce_mean(tf.square(yhat2 - y2))
e3 = tf.reduce_mean(tf.square(yhat3 - y3))
print("e2=" + str(e2))

ec = tf.concat(0, [tf.expand_dims(e2, 0), tf.expand_dims(e3, 0)])
print('ec=' + str(ec))

e_tot = tf.reduce_mean(ec)

np.random.seed(12345)
Us = np.random.randn(10, 2)
Ys = np.random.randn(10, 2)

sess = tf.Session()
e2_val = sess.run(e2, feed_dict={U:Us, Y:Ys})
print('e2_val=' + str(e2_val))
e3_val = sess.run(e3, feed_dict={U:Us, Y:Ys})
print('e3_val=' + str(e3_val))
e_tot_val = sess.run(e_tot, feed_dict={U:Us, Y:Ys})
print('e_tot_val=' + str(e_tot_val))

sys.exit(0)
"""

np.random.seed(123456)

n_in = 2
n_hid_data = 3
n_hid = 30
n_out = 1
t_in = 2000

# the "memory" of the network, how many time steps BPTT is run for
t_mem = 20

# set the time length per batch (the effective
assert t_in % t_mem == 0
n_samps = int(t_in / t_mem)

# create some fake data for training
Utrain, Xtrain, Ytrain = create_sample_data(n_in, n_hid_data, n_out, t_in, n_samps, segment_U=True)

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
Utest,Xtest,Ytest = create_sample_data(n_in, n_hid_data, n_out, t_in_test, n_samps_test, segment_U=True)

hparams = {'rnn_type':'SRNN', 'opt_algorithm':'annealed_sgd',
           'n_train_steps':60, 'n_unit':n_hid, 'dropout':{'R':0.0, 'W':0.0},
           'lambda2':1e-1}

# build the graph that will execute for each batch
graph = tf.Graph()
with graph.as_default():

    # construct the RNN
    net = Basic_Network(n_in, hparams, n_output=n_out)

    with tf.name_scope('training'):

        # the input placeholder, contains the entire multivariate input time series for a batch
        U = tf.placeholder(tf.float32, shape=[t_mem, n_in])

        # the output placeholder, contains the desired multivariate output time series for a batch
        Y = tf.placeholder(tf.float32, shape=[t_mem, n_out])

        # the initial state placeholder, contains the initial state used at the beginning of a batch
        h = tf.placeholder(tf.float32, shape=[1, n_hid])

        # placeholder for whether or not to use dropout
        p_dropout = tf.placeholder(tf.float32)

        hnext = h

        lambda2_val = hparams['lambda2']
        l2_cost = net.l2_W(lambda2_val) + net.l2_R(lambda2_val) + net.l2_Wout(lambda2_val)

        # The forward propagation graph
        net_err_list = list()
        net_preds = list()
        for t in range(t_mem):

            # construct the input vector at time t by slicing the input placeholder U, do the same for Y
            u = tf.slice(U, [t, 0], [1, n_in])
            y = tf.slice(Y, [t, 0], [1, n_out])

            # create an op to move the network state forward one time step, given the input
            # vector u and previous hidden state h
            (hnext,), yhat = net.step((hnext,), u)

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

        if hparams['opt_algorithm'] == 'adam':
            eta = tf.train.exponential_decay(.008, t, 2000, 0.5, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=eta)

        elif hparams['opt_algorithm'] == 'annealed_sgd':
            eta = tf.train.exponential_decay(5e-2, t, 5000, 0.1, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(eta)

        grads, params = zip(*optimizer.compute_gradients(net_err))

        # grads, _ = tf.clip_by_global_norm(grads, hparams['grad_clip_norm'])

        apply_grads = optimizer.apply_gradients(zip(grads, params), global_step=t)

n_train_steps = hparams['n_train_steps']
summary_freq = 100
mean_error = 0.0

# create a random initial state to start with
h0_orig = np.random.randn(n_hid) * 1e-3
h0_orig = h0_orig.reshape([1, n_hid])
h0 = h0_orig

train_errs_per_epoch = list()
test_errs_per_epoch = list()
test_preds = None
with tf.Session(graph=graph) as session:

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
            feed_dict = {U:Uk, Y:Yk, h:h0}

            # run the session to train the model for this minibatch
            to_compute = [net_err, eta, hnext, apply_grads]
            train_error_val, eta_val, hnext_val = session.run(to_compute, feed_dict=feed_dict)[:3]

            # get the last hidden state value to use on the next minibatch
            h0 = hnext_val

            print('iter=%d, batch %d: eta=%0.6f, err=%0.6f' % (step, k, eta_val, train_error_val))
            train_errs_per_samp.append(train_error_val)

        # predict on the test set
        h0 = h0_start
        Yhat = list()
        for k in range(n_samps_test):
            Uk = Utest[k, :, :]
            Yk = Ytest[k, :, :]            
            feed_dict = {U:Uk, Y:Yk, h:h0}

            to_compute = [net_err, hnext]
            to_compute.extend(net_preds)
            compute_outputs = session.run(to_compute, feed_dict=feed_dict)
            test_error_val, hnext_val = compute_outputs[:2]
            test_preds_val = np.array(compute_outputs[2:])
            test_errs_per_samp.append(test_error_val)
            h0 = hnext_val
            Yhat.append(test_preds_val)

        # overwrite the old value of test_preds, so that it's equal to the prediction for the last optimization iteration
        test_preds = np.array(Yhat)

        train_errs_per_epoch.append(train_errs_per_samp)
        test_errs_per_epoch.append(test_errs_per_samp)

    # plot the training and test error
    train_errs_per_epoch = np.array(train_errs_per_epoch)
    test_errs_per_epoch = np.array(test_errs_per_epoch)

    print('train_errs_per_epoch.shape=' + str(train_errs_per_epoch.shape))
    print('test_errs_per_epoch.shape=' + str(test_errs_per_epoch.shape))

    plt.figure()
    
    ax = plt.subplot(2, 2, 1)
    plt.errorbar(range(n_train_steps), train_errs_per_epoch.mean(axis=1), yerr=train_errs_per_epoch.std(axis=1, ddof=1),
                 linewidth=4.0, c='r', alpha=0.7, elinewidth=2.0, ecolor='k')
    plt.axis('tight')
    plt.xlabel('Epoch')
    plt.ylabel('Training Error')
    plt.title('Training Error')
    
    ax = plt.subplot(2, 2, 2)
    plt.errorbar(range(n_train_steps), test_errs_per_epoch.mean(axis=1), yerr=test_errs_per_epoch.std(axis=1, ddof=1),
                 linewidth=4.0, c='b', alpha=0.7, elinewidth=2.0, ecolor='k')
    plt.axis('tight')
    plt.xlabel('Epoch')
    plt.ylabel('Test Error')
    plt.title('Test Error')

    print('Ytest.shape=' + str(Ytest.shape))
    print('test_preds.shape=' + str(test_preds.shape))
    ax = plt.subplot(2, 2, 3)
    plt.plot(Ytest.ravel(), test_preds.ravel(), 'go', alpha=0.7)
    plt.xlabel('Y value')
    plt.ylabel('Prediction')
    plt.axis('tight')

    Y_t = Ytest.reshape([Ytest.shape[0]*Ytest.shape[1], Ytest.shape[2]])
    Yhat_t = test_preds.reshape([test_preds.shape[0]*test_preds.shape[1], test_preds.shape[2]])
    ax = plt.subplot(2, 2, 4)
    plt.plot(Y_t.squeeze(), 'k-', linewidth=4.0, alpha=0.7)
    plt.plot(Yhat_t.squeeze(), 'r-', linewidth=4.0, alpha=0.7)
    plt.axis('tight')
    plt.xlabel('Time')
    plt.ylabel('Y(t)')
    plt.legend(['Real', 'Prediction'])
    plt.title('cc=%0.2f' % (np.corrcoef(Y_t.squeeze(), Yhat_t.squeeze())[0, 1]))

    plt.show()
