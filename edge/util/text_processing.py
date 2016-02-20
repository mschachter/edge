import numpy as np
from collections import Counter

def string_to_alphabet_indices(string):
    '''Finds the alphabet used in string and returns it along with an integer
    array that re-enodes each character in the string to its integer order in
    the alphabet'''
    counter = Counter(string)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    alphabet, _ = list(zip(*count_pairs))
    alphabet = np.array(alphabet)
    alpha_ids = dict(zip(alphabet, range(len(alphabet))))
    indices = np.array(map(alpha_ids.get, string))
    return alphabet, indices

def file_to_datasets(filename, valid_fraction = .05, test_fraction = .05,
    to_lower = True):

    with open(filename, 'r') as data_file:
        text = data_file.read()
        if to_lower:
            text = text.lower()

    alphabet, text_inds = string_to_alphabet_indices(text)

    valid_start = np.floor((1 - valid_fraction - test_fraction)*text_inds.size)
    test_start = np.floor((1 - test_fraction)*text_inds.size)

    train_data = text_inds[0:valid_start]
    valid_data = text_inds[valid_start:test_start]
    test_data = text_inds[test_start:]

    return train_data, valid_data, test_data, alphabet


def char_to_onehot(char, alphabet):
    alpha_id = np.where(alphabet == char)[0][0]
    return id_to_onehot(alpha_id, alphabet)

def id_to_onehot(alpha_id, alphabet):
    input_val = np.zeros([1, len(alphabet)], dtype=np.float32)
    input_val[0, alpha_id] = 1
    return input_val
