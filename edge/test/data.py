import numpy as np


def rescale_matrix(W, spectral_radius=1.0):
    """ Rescale the given matrix W (in place) until it's spectral radius is less than or equal to the given value. """

    max_decrease_factor = 0.90
    min_decrease_factor = 0.99
    evals,evecs = np.linalg.eig(W)
    initial_eigenvalue = np.max(np.abs(evals))
    max_eigenvalue = initial_eigenvalue
    while max_eigenvalue > spectral_radius:
        #inverse distance to 1.0, a number between 0.0 (far) and 1.0 (close)
        d = 1.0 - ((max_eigenvalue - spectral_radius) / abs(initial_eigenvalue - spectral_radius))

        decrease_factor = max_decrease_factor + d*(min_decrease_factor-max_decrease_factor)
        W *= decrease_factor
        evals,evecs = np.linalg.eig(W)
        max_eigenvalue = np.max(np.abs(evals))


def create_sample_data(ninputs, nhidden, nout, nt, nsamples, segment_U=False):

    # create input time series
    U = np.random.randn(nt, ninputs)

    # create a mixing matrix to correlate input features
    # A = np.random.randn(noutputs, noutputs)
    # U = np.dot(U, A)

    # create an input weight matrix
    Win = np.random.randn(nhidden, ninputs)

    # create a recurrent weight matrix and bias
    W = np.random.randn(nhidden, nhidden)
    b = np.random.randn(nhidden)

    # adjust the spectral radius of the weight matrix
    rescale_matrix(W, 0.95)

    # create an output weight matrix and bias
    Wout = np.random.randn(nout, nhidden)*1e-1
    Wout[Wout < np.percentile(Wout, 10)] = 0.
    bout = np.random.randn(nout)
    # get rid of negative bias in output bias
    bout[bout < np.percentile(bout, 10)] = 0.

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
