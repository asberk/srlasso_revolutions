"""
solve_utils



Author: Aaron Berk <aaronsberk@gmail.com>
Copyright © 2023, Aaron Berk, all rights reserved.
Created: 16 March 2023
"""
import numpy as np
import numpy.linalg as la


def lamda_sr_best(m):
    from scipy.stats import norm

    return 1.1 * norm.ppf(1 - 0.05 / (2 * m))


def support(x_bar, A, b, lamda, y_bar):
    if not is_inexact_solution(A, b, x_bar):
        raise ValueError(
            "fancy_get_support is not designed for x_bar such that A @ x_bar = b"
        )
    J = equicorrelation(A, y_bar, lamda)
    supp = J * (~np.isclose(x_bar, 0))
    supp = supp.ravel()
    mask = ~supp
    return supp, mask


def is_inexact_solution(A, b, x_bar):
    """Checks that A @ x_bar ≠ b"""
    return not np.allclose(b, A.dot(x_bar))


def is_submatrix_injective(A, columns):
    """Checks that A[:, columns] is injective i.e. has full column rank"""
    max_rank = columns.size
    if max_rank > A.shape[0]:
        return False
    computed_rank = la.matrix_rank(A[:, columns])
    return computed_rank >= max_rank


def is_submatrix_ranges_b(A, b, columns):
    """Checks that b in Range(A[:, columns])"""
    if np.allclose(b, 0):
        return True
    beta, _, rank, sing_vals = la.lstsq(A[:, columns], b, rcond=None)
    return np.allclose(b, A[:, columns].dot(beta))


def is_uniqueness_sufficiency(
    A, b, lamda, x_bar, y_bar, aux_value, verbose=False
):
    if aux_value >= lamda:
        return False
    if not is_inexact_solution(A, b, x_bar):
        return False  # N/A!
    supp, mask = support(x_bar, A, b, lamda, y_bar)
    if supp.sum() == 0:
        return True

    max_rank = supp.sum()
    rank = la.matrix_rank(A[:, supp])
    if rank < max_rank:
        return False

    if is_submatrix_ranges_b(A, b, supp):
        return False

    Z0 = la.norm(A[:, mask].T.dot(y_bar), np.inf)
    lamda_empir = la.norm(A.T.dot(y_bar), np.inf)
    lamda_ = np.minimum(lamda, lamda_empir)

    J = equicorrelation(A, y_bar, lamda)
    if supp.sum() == J.sum():
        if np.isclose(Z0, lamda_):
            return False  # contradicts |I| = |J|!
        return True
    ZZ = np.minimum(Z0, lamda_)
    if np.isclose(aux_value, ZZ) or (aux_value >= ZZ):
        return False
    return True


def equicorrelation(A, y_bar, lamda):
    q = np.abs(A.T.dot(y_bar))
    J = np.isclose(q, lamda)
    return J


def _column_vectorize(vec):
    if (vec.ndim == 2) and vec.shape[1] == 1:
        return vec
    if np.isscalar(vec) or (vec.ndim > 2):
        return vec
    if vec.ndim == 1:
        vec = vec.reshape(-1, 1)
    if vec.shape[0] == 1:
        return vec.T
    return vec


def mse(x, x0):
    h = _column_vectorize(x) - _column_vectorize(x0)
    return (h**2).mean(axis=0)


def generate_data(m, n, s, gamma=None, seed=None):
    if gamma is None:
        gamma = 0.1
    if seed:
        np.random.seed(seed)
    A = np.random.randn(m, n) / m**0.5
    x0 = np.zeros(n)
    x0[:s] = m + m**0.5 * np.random.randn(s)
    b = A @ x0 + gamma * np.random.randn(m)
    return A, b, x0
