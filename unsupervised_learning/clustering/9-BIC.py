#!/usr/bin/env python3
"""
This module contains a function that finds the best number of clusters
for a GMM using the Bayesian Information Criterion
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using 
    the Bayesian Information Criterion (BIC)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    best_k = None
    best_result = None
    log_likelihoods = []
    bics = []

    for k in range(kmin, kmax + 1):
        try:
            pi, m, S, g, ll = expectation_maximization(X, k, iterations, tol, verbose)
            if ll is None or not np.isfinite(ll):
                continue
            if not all(np.isfinite(arr).all() for arr in [pi, m, S]):
                continue

            cov_params = d * (d + 1) / 2
            p = (k - 1) + k * d + k * cov_params
            bic = p * np.log(n) - 2 * ll

            log_likelihoods.append(ll)
            bics.append(bic)

            if best_k is None or bic < bics[best_k - kmin]:
                best_k = k
                best_result = (pi, m, S)

        except Exception:
            continue

    if best_k is None:
        return None, None, None, None

    return best_k, best_result, np.array(log_likelihoods), np.array(bics)
