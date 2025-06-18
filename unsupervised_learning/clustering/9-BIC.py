#!/usr/bin/env python3
"""This module contains a function that finds the best number
of clusters for a GMM using the Bayesian Information Criterion"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the BIC.

    Returns:
    - best_k: best number of clusters (int)
    - best_result: tuple (pi, m, S) for best k
    - log_likelihoods: np.ndarray of log likelihoods for each k
    - bics: np.ndarray of BIC values for each k
    """

    # Input validation
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None

    if kmax is None:
        kmax = kmin

    if not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    best_k = None
    best_result = None
    log_likelihoods = []
    bics = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(X, k, iterations, tol, verbose)
        if ll is None:
            return None, None, None, None

        # Number of parameters for full covariance GMM
        p = k - 1 + k * d + k * d * (d + 1) / 2
        bic = p * np.log(n) - 2 * ll

        log_likelihoods.append(ll)
        bics.append(bic)

        # Update best model if current BIC is lower (better)
        if best_k is None or bic < min(bics):
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, np.array(log_likelihoods), np.array(bics)
