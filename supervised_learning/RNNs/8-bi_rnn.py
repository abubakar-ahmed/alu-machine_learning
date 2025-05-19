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
        bi_cell: instance of BidirectionalCell
        X: ndarray (t, m, i) - input data
        h_0: ndarray (m, h) - initial forward hidden state
        h_T: ndarray (m, h) - initial backward hidden state

    Returns:
        H: ndarray (t, 2, m, h) - all hidden states
        Y: ndarray (t, m, o) - all outputs
    '''
    t, m, i = X.shape
    _, h = h_0.shape

    Hf = np.zeros((t + 1, m, h))
    Hb = np.zeros((t + 1, m, h))
    Hf[0] = h_0
    Hb[t] = h_T

    # Forward pass
    for step in range(t):
        Hf[step + 1] = bi_cell.forward(Hf[step], X[step])

    # Backward pass
    for step in reversed(range(t)):
        Hb[step] = bi_cell.backward(Hb[step + 1], X[step])

    H = np.zeros((t, 2, m, h))
    Y = []

    for step in range(t):
        H[step, 0] = Hf[step + 1]
        H[step, 1] = Hb[step]
        y = bi_cell.output(H[step])
        Y.append(y)

    Y = np.stack(Y, axis=0)  # (t, m, o)
    return H, Y
