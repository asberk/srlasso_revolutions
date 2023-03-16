import numpy as np
import numpy.linalg as la
import pandas as pd
import cvxpy as cp

from solve_utils import (
    is_inexact_solution,
    is_uniqueness_sufficiency,
    support,
    _column_vectorize,
    mse,
)


def solve_primal(A, b, lamda):
    m, n = A.shape
    x = cp.Variable((n, 1))
    prob = cp.Problem(
        cp.Minimize(cp.norm2(cp.matmul(A, x) - b) + lamda * cp.norm1(x))
    )
    prob.solve(solver="MOSEK")
    primal_value = prob.value
    x_bar = x.value
    return primal_value, x_bar


def solve_dual(A, b, lamda):
    m, n = A.shape
    y = cp.Variable((m, 1))
    dual = cp.Problem(
        cp.Maximize(cp.scalar_product(b, y)),
        [cp.norm_inf(cp.matmul(A.T, y)) <= lamda, cp.norm2(y) <= 1],
    )
    dual.solve(solver="MOSEK")
    dual_value = dual.value
    y_bar = y.value
    return dual_value, y_bar


def solve_auxiliary(A, b, lamda, x_bar, y_bar):
    """New and improved formulation where constraint is
    [A_I, y_bar].T.dot(z) = 0"""
    m, n = A.shape
    try:
        supp, mask = support(x_bar, A, b, lamda, y_bar)
    except ValueError:
        # print("Residual too close to 0; our theory does not apply here.")
        return np.inf, np.ones_like(y_bar) * np.nan
    # supp, mask = _get_support(x_bar, supp_tol)
    A_I = A[:, supp]
    A_Ic = A[:, mask]

    z = cp.Variable((m, 1))
    objective = cp.Minimize(cp.norm_inf(cp.matmul(A_Ic.T, y_bar + z)))
    constraints = [cp.scalar_product(y_bar, z) == 0]
    if supp.sum() > 0:
        constraints += [cp.matmul(A_I.T, z) == 0]
    aux = cp.Problem(
        objective,
        constraints,
    )
    aux_value = np.inf
    z_bar = None
    try:
        aux.solve(solver="MOSEK")
        aux_value = aux.value
        z_bar = z.value
    except cp.error.SolverError:
        pass
    if z_bar is None:
        z_bar = np.ones_like(y_bar) * np.nan
    return aux_value, z_bar


def _srLASSO_uniqueness(A, b, lamda, supp_tol=1e-7, verbose=False):
    m, n = A.shape
    b = _column_vectorize(b)

    primal_value, x_bar = solve_primal(A, b, lamda)
    dual_value, y_bar = solve_dual(A, b, lamda)
    aux_value, z_bar = solve_auxiliary(A, b, lamda, x_bar, y_bar)

    is_inexact = is_inexact_solution(A, b, x_bar)
    is_strong_duality = np.isclose(primal_value, dual_value)
    uniqueness_sufficiency = is_uniqueness_sufficiency(
        A, b, lamda, x_bar, y_bar, aux_value
    )

    Z0 = la.norm(A.T.dot(y_bar), np.inf)

    if verbose:
        print("Inexact:", is_inexact)
        print("Strong duality:", is_strong_duality)
        print("Sufficiency condition:", uniqueness_sufficiency)

    return (
        is_inexact,
        is_strong_duality,
        uniqueness_sufficiency,
        primal_value,
        dual_value,
        aux_value,
        x_bar,
        y_bar,
        z_bar,
        Z0,
    )


def srLASSO_uniqueness(A, b, lamda_c, x0=None, alphas=None):
    """
    Parameters
    ----------
    A: np.ndarray (m,n)
        measurement matrix
    b: np.ndarray (m, 1)
        measurement vector
    x0: np.ndarray (n, 1)
        signal
    lamda_c: float
        centre lambda value for the lambda range
    alphas: np.ndarray (num_alphas,)
        the (un)scaled range for the lambda values

    Returns
    -------
    strong_duality: np.ndarray
        bool vector asserting primal value == dual value
    uniqueness_sufficiency: np.ndarray
        bool vector asserting norm_inf(A[:, mask].T (y_bar + z_bar)) < lamda
    primal_values: np.ndarray
        optimal value for each lamda
    dual_values: np.ndarray
        dual optimal value for each lamda
    aux_values: np.ndarray
        norm_inf(A[:, mask].T (y_bar + z_bar)) for each lamda
    lamdas: np.ndarray
        vector of lamda values
    x_bars: np.ndarray (n, lamdas.size)
        matrix of solutions for each lamda
    y_bars: np.ndarray (m, lamdas.size)
        matrix of dual solutions for each lamda
    z_bars: np.ndarray (m, lamdas.size)
        matrix of aux solutions for each lamda
    """
    if alphas is None:
        alphas = np.logspace(-2, 2, 201)
    lamdas = lamda_c * alphas
    b = _column_vectorize(b)

    results = [_srLASSO_uniqueness(A, b, lamda) for lamda in lamdas]
    (
        inexact,
        strong_duality,
        uniqueness_sufficiency,
        primal_values,
        dual_values,
        aux_values,
        x_bars,
        y_bars,
        z_bars,
        Z0s,
    ) = zip(*results)

    x_bars = np.column_stack(x_bars)
    y_bars = np.column_stack(y_bars)
    z_bars = np.column_stack(z_bars)
    residuals = la.norm(A.dot(x_bars) - b.reshape(-1, 1), axis=0)

    results_df = pd.DataFrame(
        {
            "lamda": lamdas,
            "inexact": inexact,
            "strong_duality": strong_duality,
            "uniqueness_sufficiency": uniqueness_sufficiency,
            "primal_value": primal_values,
            "dual_value": dual_values,
            "aux_value": aux_values,
            "Z0": Z0s,
            "residual": residuals,
        }
    )

    norm_diffs = None
    mse_vals = None
    if x0 is not None:
        norm_diffs = la.norm(x_bars - x0.reshape(-1, 1), axis=0)
        mse_vals = mse(x_bars, x0)
        results_df["error_value"] = norm_diffs
        results_df["mse_value"] = mse_vals

    return (
        results_df,
        x_bars,
        y_bars,
        z_bars,
    )
