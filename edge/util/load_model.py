import tensorflow as tf
import yaml
import h5py
import numpy as np
import sys, os
import argparse
import dateutil

from edge.networks import Basic_Network


def find_matching_model_dir(args, use_newest = True):

    possible_dirs = []
    earliest_date = None
    earliest_dir = None

    for run_dir in os.listdir('runs'):
        with open(os.path.join('runs', run_dir, 'params.yaml')) as f:
            dir_params = yaml.load(f)

            if all(d1[k] == d2[k] for k in d1 if d1[k] is not None):
                possible_dirs.append(run_dir)
                dir_date = dateutil.parser.parse(dir_params['start_date'])

                if earliest_date is None or dir_date < earliest_date:
                    earlies_date = dir_date
                    earliest_dir = run_dir

    if len(possible_dirs) < 1:
        raise RuntimeError('There are no runs that match your specification!')
    elif len(possible_dirs) == 1:
        return possible_dirs[0]
    elif len(possible_dirs) > 1:
        if use_newest:
            return earliest_dir
        else:
            raise RuntimeError('There are multiple runs that match your specification'
                            + 'Pick one:' + str(possible_dirs))

parser = argparse.ArgumentParser('Loads saved model from disk')
parser.add_argument('--save_dir', type=str,
                   help='directory containing a saved model')
parser.add_argument('--rnn_type', type=str,
                   help='type of rnn-layer in the network')
parser.add_argument('--use_newest', action='store_true')
args = parser.parse_args()

if args.save_dir is not None:
    model_dir = args.save_dir
else:


    args = vars(args)

    use_newest = args['use_newest']
    del args['use_newest']

    model_dir = find_matching_model_dir(args, use_newest)

    print model_dir

    #net = Basic_Network(hparams['n_alphabet'], hparams)
