
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from edge.train_multivariate import read_config, MultivariateRNNTrainer


def make_training_data(nbatches=10, nsegs_per_batch=200, t_mem=50, sample_rate=1e3, freq=1.,
                       input_noise_std=1., output_noise_std=1e-2):

    nt = nsegs_per_batch*t_mem

    # generate random noise as a single channel of input
    U = np.random.randn(nbatches, nsegs_per_batch, t_mem, 1) * input_noise_std

    t = np.arange(nt) / sample_rate

    Y = np.zeros([nbatches, nsegs_per_batch, t_mem, 1])

    for b in range(nbatches):
        y = np.sin(2*np.pi*t*freq) + np.random.randn(nt)*output_noise_std
        Y[b, :, :, 0] = y.reshape([nsegs_per_batch, t_mem])

    return U,Y


def train_network():

    Utrain,Ytrain = make_training_data()
    Utest,Ytest = make_training_data(nbatches=1)
    nb,nspb,tm,nc = Utest.shape
    print 'Utest.shape=',Utest.shape
    Utest = Utest.reshape([nspb, tm, nc])
    Ytest = Ytest.reshape([nspb, tm, nc])

    print 'Utrain.shape=',Utrain.shape

    n_in = 1
    n_out = 1
    hparams = read_config('param/deep_ei.yaml', n_in, n_out, override_params={'ei_ratio':0.3})

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


if __name__ == '__main__':
    train_network()

