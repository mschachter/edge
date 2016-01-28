import numpy as np

class Input_Generator(object):
    def __init__(self, text, alphabet, n_batch, n_bptt):
        '''
        text is an array of integer character IDs and representing a text seq
        alphabet is array of characters ordered by ID
        n_batch is the number of simultaneous running inputs sequences
        n_bptt is the number time steps of forward/backprop we need input for
        '''
        self.text = text
        self.n_char = text.size
        self.n_batch = n_batch
        self.n_bptt = n_bptt
        self.n_alpha = alphabet.size

        batch_offset = self.n_char//n_batch

        self.cursors = [i*batch_offset for i in xrange(n_batch)]
        self.last_batch = self.next_batch()

    def next_batch(self):
        batch = np.zeros((self.n_batch, self.n_alpha), dtype=np.float32)
        for i in xrange(self.n_batch):
            batch[i, self.text[self.cursors[i]]] = 1.0
            self.cursors[i] = (self.cursors[i] + 1) % self.n_char
        return batch

    def next_window(self):
        ''' Returns the next sequence of batches whose length is n_bptt + 1 '''
        window = [self.last_batch]
        for i in xrange(self.n_bptt):
            window.append(self.next_batch())
        self.last_batch = window[-1]
        return window
