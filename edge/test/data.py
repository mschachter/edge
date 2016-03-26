import numpy as np


def create_sample_data(ninputs, nhidden, nout, nt, nsamples, segment_U=False):

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

    # create an output weight matrix and bias
    Wout = np.random.randn(nout, nhidden)
    bout = np.random.randn(nout)

    # random initial state
    np.random.seed(123456)
    x0 = np.random.randn(nhidden)

    # create an output matrix
    nt_per_sample = nt / nsamples
    assert nt % nsamples == 0

    if segment_U:
        Us = list()
        Xs = np.zeros([nsamples, nt_per_sample, nhidden])
        Ys = np.zeros([nsamples, nt_per_sample, nout])
        for sample_idx in range(nsamples):
            si = sample_idx*nt_per_sample
            ei = si + nt_per_sample

            xxx,yyy = run_network(U[si:ei, :], Win, W, b, Wout, bout, x0)
            Xs[sample_idx, :, :] = xxx[1:, :]
            Ys[sample_idx, :, :] = yyy

            x0 = Xs[sample_idx, -1, :]
            Us.append(U[si:ei, :])
        Us = np.array(Us)
    else:
        Us = U
        Xs,Ys = run_network(U, Win, W, b, Wout, bout, x0)
        Xs = Xs[1:, :]

    return Us, Xs, Ys


def run_network(U, Win, W, bhid, Wout, bout, x0, activation=np.tanh):

    # get the shape of everything
    nt,ninputs = U.shape
    assert Win.shape[1] == ninputs

    nhidden,nhidden2 = W.shape
    assert nhidden == nhidden2

    assert Win.shape[0] == nhidden

    assert len(x0) == nhidden

    assert len(bhid) == nhidden

    nout,nhidden3 = Wout.shape
    assert nhidden3 == nhidden

    assert len(bout) == nout

    # create a matrix to hold the network state (excluding x0)
    X = np.zeros([nt+1, nhidden])
    X[0, :] = x0

    # create a matrix to hold hte output
    Y = np.zeros([nt, nout])

    for t in range(nt):
        # compute next state
        X[t+1, :] = activation(np.dot(W, X[t, :]) + bhid + np.dot(Win, U[t, :]))
        Y[t, :] = np.dot(Wout, X[t+1, :]) + bout

    return X,Y
