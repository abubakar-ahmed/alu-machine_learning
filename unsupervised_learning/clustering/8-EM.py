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
    k: positive integer containing the number of clusters
    iterations: max number of iterations
    tol: non-negative float for tolerance of the log likelihood
    verbose: boolean that determines if output should be printed

    Returns:
        pi: numpy.ndarray of shape (k,) - priors for each cluster
        m: numpy.ndarray of shape (k, d) - centroid means for each cluster
        S: numpy.ndarray of shape (k, d, d) - covariance matrices for each cluster
        g: numpy.ndarray of shape (k, n) - posterior probabilities for each data point
        log_likelihood: log likelihood of the model
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
        not isinstance(k, int) or k <= 0 or
        not isinstance(iterations, int) or iterations <= 0 or
        not isinstance(tol, float) or tol < 0 or
        not isinstance(verbose, bool)):
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
