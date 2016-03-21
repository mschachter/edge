import tensorflow as tf
import yaml
import h5py
import numpy as np
import sys, os
import argparse
import dateutil.parser

import text_processing as text_proc
from edge.networks import Basic_Network
from edge.sampler import Sampler

def load_model_params(model_path):
    param_path = os.path.join(model_path, 'params.yaml')
    if not os.path.exists(param_path):
        return None
    with open(param_path, 'r') as f:
        hparams = yaml.load(f)
    return hparams

def find_matching_model_dir(match_params, use_newest = True):

    possible_dirs = []
    earliest_date = None
    earliest_dir = None

    for run_dir in os.listdir('runs'):
        run_dir = os.path.join('runs', run_dir)

        dir_params = load_model_params(run_dir)
        if dir_params is None:
            continue

        if all(match_params[k] == dir_params[k] for k in match_params
            if match_params[k] is not None):

            possible_dirs.append(run_dir)
            dir_date = dateutil.parser.parse(dir_params['start_date'])

            if earliest_date is None or dir_date < earliest_date:
                earlies_date = dir_date
                earliest_dir = run_dir


    if use_newest and len(possible_dirs) > 1:
        return [earliest_dir]
    else:
        return possible_dirs


    # if len(possible_dirs) < 1:
    #     raise RuntimeError('There are no runs that match your specification!')
    # elif len(possible_dirs) == 1:
    #     return possible_dirs[0]
    # if len(possible_dirs) > 1 & use:
    #     import ipdb; ipdb.set_trace()
    #     if use_newest:
    #         return earliest_dir
    #     else:
    #         raise RuntimeError('There are multiple runs that match your specification'
    #                         + 'Pick one:' + str(possible_dirs))


if __name__ == '__main__':
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


    hparams = load_model_params(model_dir)

    net = Basic_Network(hparams['n_alphabet'], hparams)

    saver = tf.train.Saver()
    session = tf.InteractiveSession()
    saver.restore(session, os.path.join(model_dir, 'model.ckpt'))

    data_path = os.path.join(hparams['data_dir'], hparams['data_file'])

    train_text, valid_text, test_text, alphabet = \
        text_proc.file_to_datasets(data_path)
    n_alphabet = len(alphabet)

    sampler = Sampler(net, alphabet)
