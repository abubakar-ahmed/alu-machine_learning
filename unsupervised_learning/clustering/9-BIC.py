#!/usr/bin/env python3
"""This module contains a function that finds
the best number of clusters for a GMM using the
Bayesian Information Criterion (BIC).
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the BIC.

    Parameters:
    - X: np.ndarray of shape (n, d) containing the data set.
    - kmin: Minimum number of clusters to check for (inclusive).
    - kmax: Maximum number of clusters to check for (inclusive).
    - iterations: Maximum number of iterations for EM.
    - tol: Tolerance for convergence.
    - verbose: Boolean that determines if logs should be printed.

    Returns:
    - best_k: The best value for k (number of clusters).
    - best_result: Tuple containing the best (pi, m, S).
    - lls: List of log likelihoods for each tested k.
    - bics: List of BICs for each tested k.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None, None, None
    if not isinstance(iterations, int):
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    if kmax is None:
        kmax = kmin

    n, d = X.shape
    best_k = None
    best_result = None
    best_bic = float('inf')

    likelyhoods = []
    bics = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(X, k, iterations, tol, verbose)

        # Number of parameters:
        # Means: k * d
        # Covariances: k * d * (d + 1) / 2 (symmetric matrices)
        # Priors: k - 1 (since they sum to 1)
        p = k * d + k * d * (d + 1) / 2 + (k - 1)
        bic = p * np.log(n) - 2 * ll

        likelyhoods.append(ll)
        bics.append(bic)

        if bic < best_bic:
            best_k = k
            best_result = (pi, m, S)
            best_bic = bic

    return best_k, best_result, np.array(likelyhoods), np.array(bics)
