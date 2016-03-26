from __future__ import division

import numpy as np
import sys

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
n_hid = 3
n_out = 1
t_in = 10

# the "memory" of the network, how many time steps BPTT is run for
t_mem = 2

# set the time length per batch (the effective
assert t_in % t_mem == 0
n_samps = int(t_in / t_mem)

# create some fake data
Us,Xs,Ys = create_sample_data(n_in, n_hid, n_out, t_in, n_samps, segment_U=True)
hparams = {'rnn_type':'SRNN', 'opt_algorithm':'annealed_sgd', 'n_train_steps':3, 'n_unit':n_hid}

# build the graph that will execute for each batch
graph = tf.Graph()
with graph.as_default():

    # construct the RNN
    net = Basic_Network(n_in, hparams)

    with tf.name_scope('training'):

        # the input placeholder, contains the entire multivariate input time series for a batch
        U = tf.placeholder(tf.float32, shape=[t_mem, n_in])

        # the output placeholder, contains the desired multivariate output time series for a batch
        Y = tf.placeholder(tf.float32, shape=[t_mem, n_out])

        # the initial state placeholder, contains the initial state used at the beginning of a batch
        h = tf.placeholder(tf.float32, shape=[1, n_hid])
        hnext = h

        # The forward propagation graph
        errs = list()
        for t in range(t_mem):

            # construct the input vector at time t by slicing the input placeholder U, do the same for Y
            u = tf.slice(U, [t, 0], [1, n_in])
            y = tf.slice(Y, [t, 0], [1, n_out])

            # create an op to move the network state forward one time step, given the input
            # vector u and previous hidden state h
            print('t=%d' % t)
            (hnext,), yhat = net.step((hnext,), u)

            # compute the cost at time t
            err = tf.reduce_mean(tf.squeeze(tf.square(y - yhat)))
            errs.append(tf.expand_dims(err, 0))

        # compute the overall training error, the mean of the mean square error at each time point
        print('errs=')
        print(errs)
        errs = tf.concat(0, errs)
        print('errs2=')
        print(errs)
        train_err = tf.reduce_mean(errs)

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

        # grads, _ = tf.clip_by_global_norm(grads, hparams['grad_clip_norm'])

        apply_grads = optimizer.apply_gradients(zip(grads, params), global_step=t)

n_train_steps = hparams['n_train_steps']
summary_freq = 100
mean_error = 0.0

# create a random initial state to start with
h0 = np.random.randn(n_hid) * 1e-3
h0 = h0.reshape([1, n_hid])

with tf.Session(graph=graph) as session:

    tf.initialize_all_variables().run()

    # for each training iteration
    for step in range(n_train_steps):

        # train on each minibatch
        for k in range(n_samps):
            print("step=%d, k=%d" % (step, k))
            # put the input and output matrices in the feed dictionary for this minibatch
            Uk = Us[k, :, :]
            Yk = Ys[k, :, :]
            print('Uk.shape=' + str(Uk.shape))
            print('Yk.shape=' + str(Yk.shape))
            print("h0.shape=" + str(h0.shape))
            feed_dict = {U:Uk, Y:Yk, h:h0}

            # run the session to train the model for this minibatch
            to_compute = [train_err, eta, hnext, apply_grads]
            error_val, eta_val, hnext_val = session.run(to_compute, feed_dict=feed_dict)[:3]

            # get the last hidden state value to use on the next minibatch
            h0 = hnext_val

            print('iter=%d, batch %d: eta=%0.6f, err=%0.6f' % (step, k, eta_val, error_val))
            print('hnext_val=')
            print(hnext_val)
