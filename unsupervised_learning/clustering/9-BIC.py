#!/usr/bin/env python3
"""This module contains a function that finds the best number
of clusters for a GMM using the Bayesian Information Criterion"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the
    Bayesian Information Criterion

    Parameters:
    - X: np.ndarray of shape (n, d) containing the data set
    - kmin: minimum number of clusters to check (int)
    - kmax: maximum number of clusters to check (int)
    - iterations: max iterations for EM algorithm (int)
    - tol: tolerance for EM algorithm convergence (float)
    - verbose: boolean to print info during EM

    Returns:
    - best_k: best number of clusters (int)
    - best_result: tuple (pi, m, S) for best k
    - log_likelihoods: np.ndarray of log likelihoods for each k
    - bics: np.ndarray of BIC values for each k
    """

    # Validate input types and values
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None

    if kmax is None:
        kmax = kmin  # default to at least kmin

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
        # Run EM algorithm for k clusters
        pi, m, S, g, ll = expectation_maximization(X, k, iterations, tol, verbose)

        # If EM failed to converge or returned None log likelihood
        if ll is None:
            return None, None, None, None

        # Number of parameters p for GMM with full covariance:
        # p = k - 1 (weights) + k*d (means) + k*d*(d+1)/2 (covariances)
        p = k - 1 + k * d + k * d * (d + 1) / 2

        # Compute BIC for this k
        bic = p * np.log(n) - 2 * ll

        # Save likelihood and BIC
        log_likelihoods.append(ll)
        bics.append(bic)

        # Track best BIC (lowest is better)
        if best_k is None or bic < min(bics):
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, np.array(log_likelihoods), np.array(bics)
