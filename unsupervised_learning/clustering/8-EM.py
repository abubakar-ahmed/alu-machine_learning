#!/usr/bin/env python3

"""
This module contains a function that performs
expectation maximization for a GMM
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM

    X: numpy.ndarray of shape (n, d) containing the dataset
        - n: number of data points
        - d: number of dimensions of each data point
    k: positive integer containing the number of clusters
    iterations: positive integer containing the maximum number of iterations
    tol: non-negative float containing tolerance of the log likelihood
    verbose: boolean that determines if output should be printed

    Returns:
        pi: numpy.ndarray of shape (k,) containing the priors for each cluster
        m: numpy.ndarray of shape (k, d) containing the centroid means for each cluster
        S: numpy.ndarray of shape (k, d, d) containing the covariance matrices for each cluster
        g: numpy.ndarray of shape (k, n) containing the posterior probabilities
        log_likelihood: log likelihood of the model
    """
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
    g, log_likelihood = expectation(X, pi, m, S)
    prev_like = log_likelihood
    msg = "Log Likelihood after {} iterations: {}"

    if verbose:
        print(msg.format(0, log_likelihood.round(5)))

    for i in range(iterations):
        pi, m, S = maximization(X, g)
        g, log_likelihood = expectation(X, pi, m, S)

        if verbose and (i + 1) % 10 == 0:
            print(msg.format(i + 1, log_likelihood.round(5)))

        if abs(prev_like - log_likelihood) <= tol:
            break

        prev_like = log_likelihood

    if verbose and (i + 1) % 10 != 0:
        print(msg.format(i + 1, log_likelihood.round(5)))

    return pi, m, S, g, log_likelihood
