#!/usr/bin/env python3
'''
Expectation Maximization for a Gaussian Mixture Model
'''

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization

def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    '''
    Performs the Expectation Maximization algorithm for a GMM using a single loop
    '''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    l_prev = 0

    for i in range(iterations):
        # E-step
        g, l = expectation(X, pi, m, S)

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {:.5f}".format(i, l))

        if abs(l - l_prev) <= tol:
            if verbose:
                print("Log Likelihood after {} iterations: {:.5f}".format(i, l))
            break

        # M-step
        pi, m, S = maximization(X, g)
        l_prev = l

    return pi, m, S, g, l
