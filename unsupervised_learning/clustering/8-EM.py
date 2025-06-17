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
    Performs the Expectation Maximization algorithm for a GMM
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

    # Step 1: Initialization
    pi, m, S = initialize(X, k)
    g, l_prev = expectation(X, pi, m, S)

    if verbose:
        print(f"Log Likelihood after 0 iterations: {l_prev:.5f}")

    for i in range(1, iterations + 1):
        # Step 2: M-step
        pi, m, S = maximization(X, g)

        # Step 3: E-step
        g, l = expectation(X, pi, m, S)

        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {l:.5f}")

        # Step 4: Check for convergence
        if abs(l - l_prev) <= tol:
            if verbose:
                print(f"Log Likelihood after {i} iterations: {l:.5f}")
            break

        l_prev = l

    return pi, m, S, g, l
