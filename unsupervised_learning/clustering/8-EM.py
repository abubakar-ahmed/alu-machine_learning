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
    prev_log_likelihood = 0

    for i in range(iterations):
        # E-step
        g, log_likelihood = expectation(X, pi, m, S)

        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {log_likelihood:.5f}")

        if abs(log_likelihood - prev_log_likelihood) <= tol:
            # Print at last iteration if not printed already
            if verbose and (i % 10 != 0):
                print(f"Log Likelihood after {i} iterations: {log_likelihood:.5f}")
            break

        # M-step
        pi, m, S = maximization(X, g)
        prev_log_likelihood = log_likelihood

    else:
        # If loop didn't break early and finished all iterations
        if verbose and (iterations - 1) % 10 != 0:
            print(f"Log Likelihood after {iterations - 1} iterations: {log_likelihood:.5f}")

    return pi, m, S, g, log_likelihood
