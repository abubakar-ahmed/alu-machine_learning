#!/usr/bin/env python3
'''
    Script that defines a function def bi_rnn(bi_cell, X, h_0, h_t):
    that performs forward propagation for a bidirectional RNN:
'''

import numpy as np

def bi_rnn(bi_cell, X, h_0, h_T):
    '''
        Performs forward propagation for a bidirectional RNN

        Parameters:
        - bi_cell: an instance of BidirectionalCell
        - X: numpy.ndarray of shape (t, m, i)
        - h_0: numpy.ndarray of shape (m, h) for forward direction
        - h_T: numpy.ndarray of shape (m, h) for backward direction

        Returns:
        - H: numpy.ndarray of shape (t, m, 2 * h), concatenated hidden states
        - Y: numpy.ndarray of shape (t, m, o), outputs
    '''
    t, m, i = X.shape
    h = h_0.shape[1]
    
    # Forward pass
    H_forward = np.zeros((t + 1, m, h))
    H_forward[0] = h_0
    for step in range(t):
        H_forward[step + 1] = bi_cell.forward(H_forward[step], X[step])
    
    # Backward pass
    H_backward = np.zeros((t + 1, m, h))
    H_backward[t] = h_T
    for step in range(t - 1, -1, -1):
        H_backward[step] = bi_cell.backward(H_backward[step + 1], X[step])
    
    # Concatenate forward and backward hidden states
    H = np.concatenate((H_forward[1:], H_backward[:-1]), axis=-1)
    
    # Compute outputs - determine output size dynamically
    first_output = bi_cell.output(H[0])
    o = first_output.shape[-1]
    Y = np.zeros((t, m, o))
    Y[0] = first_output
    for step in range(1, t):
        Y[step] = bi_cell.output(H[step])
    
    return H, Y
