from __future__ import division
import numpy as np

def to_non_ed_equilavent_layer_size(n_state, n_input):

    m_state = (-n_input + np.sqrt(n_input**2 + 4*(2*n_state**2 + n_input*n_state)))/2
    return m_state

def to_ed_equilavent_layer_size(n_state, n_input):

    m_state = (-n_input + np.sqrt(n_input**2 + 4*(n_state**2 + n_input*n_state)))/2
    return m_state
