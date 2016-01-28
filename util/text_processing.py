import numpy as np


def string_to_indices(string):
    alphabet = np.unique(list(string))
    chars = np.array(list(string))
    indices = np.zeros(len(chars), dtype = np.uint8)
    for i in xrange(0, alphabet.size-1):
        indices[chars == alphabet[i]] = i
    return indices, alphabet

def file_to_datasets(filename, valid_fraction = .05, test_fraction = .05,
    to_lower = True):

    with open(filename, 'r') as data_file:
        text = data_file.read()
        if to_lower:
            text = text.lower()

    text_inds, alphabet = string_to_indices(text)

    valid_start = np.floor((1 - valid_fraction - test_fraction)*text_inds.size)
    test_start = np.floor((1 - test_fraction)*text_inds.size)

    train_data = text_inds[0:valid_start]
    valid_data = text_inds[valid_start:test_start]
    test_data = text_inds[test_start:]

    return train_data, valid_data, test_data, alphabet
