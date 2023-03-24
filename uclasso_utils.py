"""
uclasso_utils



Author: Aaron Berk <aaronsberk@gmail.com>
Copyright © 2023, Aaron Berk, all rights reserved.
Created: 24 March 2023
"""
import numpy as np
import numpy.linalg as la


def equicorrelation(A, r_bar, lamda, as_set=False):
    q = np.abs(A.T.dot(r_bar))
    J1 = np.isclose(q, lamda).ravel()
    J2 = q >= lamda
    J = J1 | J2
    if not as_set:
        return J
    return set(np.where(J)[0])


def get_support(x_bar, A, b, lamda, use_equicorrelation=False):
    r_bar = b.ravel() - A.dot(x_bar).ravel()
    supp = ~np.isclose(x_bar, 0).ravel()
    if use_equicorrelation:
        J = equicorrelation(A, r_bar, lamda)
        supp = J * supp
    mask = ~supp
    return supp, mask


def lipschitz_bound_lamda(x_bar, A, b, lamda):
    assert x_bar.size == np.max(x_bar.shape)
    x_bar = x_bar.ravel()
    support, mask = get_support(x_bar, A, b, lamda)
    s_ = support.sum()

    if s_ < 1:
        return 0.0

    AI = A[:, support]
    r = A.dot(x_bar).ravel() - b.ravel()

    rho, resid, rank, svdvals = la.lstsq(AI, r, rcond=None)
    if rank != s_:
        print(f"A_I rank-deficient. rank: {rank} sparsity: {s_}.")
        return np.nan

    out = la.norm(rho) / lamda
    return out
