import numpy as np

class Input_Generator(object):
    def __init__(self, text, alphabet, n_batch, n_prop):
        '''
        text is an array of integer character IDs and representing a text seq
        alphabet is array of characters ordered by ID
        n_batch is the number of simultaneous running inputs sequences
        n_prop is the number time steps of forward/backprop we need input for
        '''
        self.text = text
        self.n_char = text.size
        self.n_batch = n_batch
        self.n_prop = n_prop
        self.n_alpha = len(alphabet)

        self.cursors = None
        self.restart_cursors()

        self.last_batch = self.next_batch()

    def restart_cursors(self):
        batch_offset = self.n_char//self.n_batch
        self.cursors = [i*batch_offset for i in xrange(self.n_batch)]

    def next_batch(self):
        batch = np.zeros((self.n_batch, self.n_alpha), dtype=np.float32)
        for i in xrange(self.n_batch):
            batch[i, self.text[self.cursors[i]]] = 1.0
            self.cursors[i] = (self.cursors[i] + 1) % self.n_char
        return batch

    def next_window(self):
        ''' Returns the next sequence of batches whose length is n_prop + 1 '''
        window = [self.last_batch]
        for i in xrange(self.n_prop):
            window.append(self.next_batch())
        self.last_batch = window[-1]
        return window
