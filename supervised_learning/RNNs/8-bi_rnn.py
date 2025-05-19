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
    _, h = h_0.shape

    # Forward direction
    H_f = np.zeros((t + 1, m, h))
    H_f[0] = h_0
    for step in range(t):
        H_f[step + 1] = bi_cell.forward(H_f[step], X[step])

    # Backward direction
    H_b = np.zeros((t + 1, m, h))
    H_b[t] = h_T
    for step in reversed(range(t)):
        H_b[step] = bi_cell.backward(H_b[step + 1], X[step])

    # Concatenate hidden states and compute output
    H = np.zeros((t, m, 2 * h))
    Y = []
    for step in range(t):
        H[step] = np.concatenate((H_f[step + 1], H_b[step]), axis=1)
        Y.append(bi_cell.output(H[step]))

    Y = np.array(Y)
    return H, Y
