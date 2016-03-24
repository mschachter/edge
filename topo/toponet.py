import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import SimpleRNN, Dense

from keras.layers.core import Activation
from keras.models import Sequential


def create_sample_data(ninputs, nhidden, nt, nsamples):

    # create input time series
    U = np.random.randn(nt, ninputs)

    # create a mixing matrix to correlate input features
    # A = np.random.randn(noutputs, noutputs)
    # U = np.dot(U, A)

    # create an input weight matrix
    Win = np.random.randn(nhidden, ninputs)

    # create a topologically organized recurrent weight matrix and bias
    W = np.random.randn(nhidden, nhidden)
    b = np.random.randn(nhidden)

    # random initial state
    np.random.seed(123456)
    x0 = np.random.randn(nhidden)

    # create an output matrix
    nt_per_sample = nt / nsamples
    assert nt % nsamples == 0

    Us = list()
    Xs = np.zeros([nsamples, nt_per_sample, nhidden])
    for sample_idx in range(nsamples):
        si = sample_idx*nt_per_sample
        ei = si + nt_per_sample
        xxx = run_network(U[si:ei, :], Win, W, b, x0)
        Xs[sample_idx, :, :] = xxx[1:, :]
        x0 = Xs[sample_idx, -1, :]
        Us.append(U[si:ei, :])

    return np.array(Us), Xs


def run_network(U, Win, W, bhid, x0, activation=np.tanh):

    # get the shape of everything
    nt,ninputs = U.shape
    assert Win.shape[1] == ninputs

    nhidden,nhidden2 = W.shape
    assert nhidden == nhidden2

    assert Win.shape[0] == nhidden

    assert len(x0) == nhidden

    assert len(bhid) == nhidden

    # create a matrix to hold the network state (excluding x0)
    X = np.zeros([nt+1, nhidden])
    X[0, :] = x0

    for t in range(nt):
        # compute next state
        X[t+1, :] = activation(np.dot(W, X[t, :]) + bhid + np.dot(Win, U[t, :]))

    return X


if __name__ == '__main__':

    ninput = 5
    nhidden = 10
    Us,Xs = create_sample_data(ninput, nhidden, 1000, 10)
    nsamps,nt_per_sample,nfeatures = Us.shape

    net = Sequential()
    # net.add(Dense(nhidden, input_dim=ninput, init='uniform'))
    # net.add(SimpleRNN(nhidden, stateful=True, batch_input_shape=(Us.shape)))
    net.add(SimpleRNN(nhidden, stateful=False, init='uniform', input_shape=(nt_per_sample, ninput)))
    net.add(Activation('tanh'))

    net.compile(loss='mean_squared_error', optimizer='rmsprop')

    bsize = Us.shape[0]
    net.fit(Us, Xs, batch_size=bsize, nb_epoch=100, verbose=True, show_accuracy=False)
    score = net.evaluate(Us, Xs, batch_size=bsize)
