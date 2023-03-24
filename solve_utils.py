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


def support(x_bar, A, b, lamda, y_bar=None):
    if not is_inexact_solution(A, b, x_bar):
        raise ValueError(
            "fancy_get_support is not designed for x_bar such that A @ x_bar = b"
        )
    if y_bar is None:
        y_bar = b.ravel() - A.dot(x_bar).ravel()
        J = equicorrelation(A, y_bar, la.norm(y_bar) * lamda)
    else:
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


def lipschitz_bound_b_lamda(x_bar, A, b, lamda, y_bar):
    """NOT USED"""
    assert x_bar.size == np.max(x_bar.shape)
    x_bar = x_bar.ravel()
    supp, mask = support(x_bar, A, b, lamda, y_bar)
    s_ = supp.sum()

    if s_ < 1:
        return 0.0

    AI = A[:, supp]

    r = A.dot(x_bar) - b
    v = r / la.norm(r)

    AIdagv, _, rank, singvals = la.lstsq(AI, v, rcond=None)

    if rank != s_:
        print(
            f"Warning: A_I is a rank-deficient matrix. rank: {rank} s_: {s_}."
        )
        return np.nan

    sigma_min = singvals[-1]
    sigma_max2 = la.svd(
        AI.T.dot(np.eye(v.size) - v.reshape(-1, 1).dot(v.reshape(1, -1))),
        compute_uv=False,
    )[0]
    AITv = AI.T.dot(v)
    vTAIAIdagv = v.dot(AI).dot(AIdagv)
    quantity1 = (
        AITv.dot(AITv) * (1 + vTAIAIdagv) / (1 - vTAIAIdagv**2)
        + 1 / sigma_min**2
    )
    quantity2 = sigma_max2 + la.norm(AI.T.dot(r) / lamda)

    return quantity1 * quantity2


def lipschitz_bound_lamda(x_bar, A, b, lamda, y_bar=None):
    assert x_bar.size == np.max(x_bar.shape)
    x_bar = x_bar.ravel()
    supp, mask = support(x_bar, A, b, lamda, y_bar)
    s_ = supp.sum()

    if s_ < 1:
        return 0.0

    AI = A[:, supp]

    r = A.dot(x_bar) - b
    R = la.norm(r)
    v = r / R

    AIdagv, _, rank, singvals = la.lstsq(AI, v, rcond=None)

    if rank != s_:
        print(
            f"Warning: A_I is a rank-deficient matrix. rank: {rank} s_: {s_}."
        )
        return np.nan

    vTAIAIdagv = v.dot(AI).dot(AIdagv)

    out = la.norm(AIdagv) * R / lamda / np.abs(1 - vTAIAIdagv)
    return out


def lipschitz_bound_ratio(x_bar, A, b, lamda, y_bar):
    """Compute the quantity
    |1 - v^T.A_I.A_I^dagger.v|
    where v = r/norm(r) with r = b - A.x_bar.
    Assuming x_bar_sr approx x_bar_uc and I_sr approx I_uc, this quantity is the
    "discrepancy" between the UC and SR local Lipschitz bounds.
    """
    supp, _ = support(x_bar, A, b, lamda, y_bar)
    AI = A[:, supp]
    rho, _, rank, svdvals = la.lstsq(AI, y_bar.ravel(), rcond=None)
    V = y_bar.dot(AI.dot(rho))
    return np.abs(1 - V)
