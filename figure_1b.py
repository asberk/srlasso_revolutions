"""
figure_1b



Author: Aaron Berk <aaronsberk@gmail.com>
Copyright © 2023, Aaron Berk, all rights reserved.
Created: 24 March 2023
"""


import os
from glob import glob
import numpy as np
import numpy.linalg as la

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path

import solve_utils as sr
import uclasso_utils as uc


def _srLASSO(A, b, lamda, verbose=False):
    m, n = A.shape
    b = sr._column_vectorize(b)

    primal_value, x_bar = sr._solve_primal(A, b, lamda)
    dual_value, y_bar = sr._solve_dual(A, b, lamda)

    is_strong_duality = np.isclose(primal_value, dual_value)
    return (
        is_strong_duality,
        primal_value,
        dual_value,
        x_bar,
        y_bar,
    )


def srLASSO(A, b, lamda_c, x0=None, alphas=None):
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
    (results_df, x_bars, y_bars)
    """
    if alphas is None:
        alphas = np.logspace(-2, 2, 51)
    lamdas = lamda_c * alphas
    b = sr._column_vectorize(b)

    results = [_srLASSO(A, b, lamda) for lamda in lamdas]
    (
        strong_duality,
        primal_values,
        dual_values,
        x_bars,
        y_bars,
    ) = zip(*results)

    x_bars = np.column_stack(x_bars)
    y_bars = np.column_stack(y_bars)
    residual_norms = la.norm(A.dot(x_bars) - b.reshape(-1, 1), axis=0)

    results_df = pd.DataFrame(
        {
            "lamda": lamdas,
            "strong_duality": strong_duality,
            "primal_value": primal_values,
            "dual_value": dual_values,
            "residual": residual_norms,
        }
    )

    norm_diffs = None
    mse_vals = None
    if x0 is not None:
        norm_diffs = la.norm(x_bars - x0.reshape(-1, 1), axis=0)
        mse_vals = sr.mse(x_bars, x0)
        results_df["error_value"] = norm_diffs
        results_df["mse_value"] = mse_vals

    return (results_df, x_bars, y_bars)


def ucLASSO(A, b, lamda_c, x0=None, alphas=None):
    if alphas is None:
        alphas = np.logspace(-2, 2, 51)
    lamdas = lamda_c * alphas
    lamdas_, x_bars, gaps = lasso_path(A, b.ravel(), alphas=lamdas)
    lamdas_ = m * lamdas_
    residual_norms = la.norm(A.dot(x_bars) - b.reshape(-1, 1), axis=0)

    results = pd.DataFrame(
        {
            "lamda": lamdas_,
            "residual": residual_norms,
        }
    )

    norm_diffs = None
    mse_vals = None
    if x0 is not None:
        norm_diffs = la.norm(x_bars - x0.reshape(-1, 1), axis=0)
        mse_vals = sr.mse(x_bars, x0)
        results["error_value"] = norm_diffs
        results["mse_value"] = mse_vals
    return (results, x_bars)


def run_trial(m, n, s, gamma, alphas=None, data_dir=None, seed=None):
    if data_dir is None:
        data_dir = "data/example_introduction_sr_uc"
    sr.mkdir(data_dir)

    A, b, x0 = sr.generate_data(m, n, s, gamma, seed)
    lamda_star_sr = sr.lamda_sr_best(m) / m**0.5
    lamda_star_uc = gamma * np.sqrt(2 * np.log(n)) / m
    tstamp = sr.get_tstamp()
    (
        results_sr,
        x_bars_sr,
        y_bars_sr,
    ) = srLASSO(A, b, lamda_star_sr, x0, alphas=alphas)
    results_sr["m"] = m
    results_sr["n"] = n
    results_sr["s"] = s
    results_sr["gamma"] = gamma

    (results_uc, x_bars_uc) = ucLASSO(A, b, lamda_star_uc, x0, alphas=alphas)
    results_uc["m"] = m
    results_uc["n"] = n
    results_uc["s"] = s
    results_uc["gamma"] = gamma

    data = [A, b, x0]

    data_string = f"{tstamp}_m{m}_n{n}_s{s}_gamma{gamma}"
    data_string = data_string.replace(".", "")
    results_sr.to_csv(os.path.join(data_dir, f"results_sr_{data_string}.csv"))
    results_uc.to_csv(os.path.join(data_dir, f"results_uc_{data_string}.csv"))
    np.savez_compressed(
        os.path.join(data_dir, f"objects_{data_string}.npz"),
        x_bars_sr,
        y_bars_sr,
        x_bars_uc,
    )
    np.savez_compressed(
        os.path.join(data_dir, f"objects2_{data_string}.npz"), *data
    )
    return results_sr, results_uc, (x_bars_sr, y_bars_sr, x_bars_uc), data


def load_trial(load_dir=None):
    """
    Returns
    -------
    dframe: pd.DataFrame
        All tabular results data in a dataframe, for trials, gammas and lamdas.
    solutions: list of (x_bars, y_bars, z_bars) tuples
        Each tuple member is a 2D np.ndarray
    solution_parms: pd.DataFrame
        The index for each tuple element in solutions corresponds with the same
        loc in solution_parms.
    """
    if load_dir is None:
        load_dir = "data/example_introduction_sr_uc"
    dframesr_fpaths = sorted(glob(os.path.join(load_dir, "results_sr*csv")))
    dframeuc_fpaths = sorted(glob(os.path.join(load_dir, "results_uc*csv")))
    npz_fpaths = sorted(glob(os.path.join(load_dir, "objects_*npz")))
    npz2_fpaths = sorted(glob(os.path.join(load_dir, "objects2_*npz")))

    def _parse_name(fpath):
        fname = os.path.basename(fpath)
        fname, ext = os.path.splitext(fname)
        fparts = fname.split("_")
        tstamp = fparts[2]
        return tstamp

    tstamps = [
        _parse_name(dframesr_fpath) for dframesr_fpath in dframesr_fpaths
    ]

    def _load_dframe_with_parms(fpath, tstamp):
        dframe = pd.read_csv(fpath, index_col=0)
        dframe["tstamp"] = tstamp
        return dframe

    def _load_npz(fpath):
        return np.load(fpath).values()

    dframesr = [
        _load_dframe_with_parms(fpath, parms)
        for fpath, parms in zip(dframesr_fpaths, tstamps)
    ]
    dframeuc = [
        _load_dframe_with_parms(fpath, parms)
        for fpath, parms in zip(dframeuc_fpaths, tstamps)
    ]
    dframe_sr = pd.concat(dframesr, axis=0)
    dframe_uc = pd.concat(dframeuc, axis=0)
    solutions = [_load_npz(fpath) for fpath in npz_fpaths][0]
    initial_data = [_load_npz(fpath) for fpath in npz2_fpaths][0]

    return dframe_sr, dframe_uc, solutions, initial_data


def compute_empirical_lipschitzness(results, x_bars):
    idx = results.mse_value.idxmin()
    lamda_star = results.lamda[idx]
    x_star = x_bars[:, idx].reshape(-1, 1)
    lipsch_empir = la.norm(x_bars - x_star, axis=0)
    dlamda = results.lamda - lamda_star
    out = pd.DataFrame({"dlamda": dlamda, "lipsch_empir": lipsch_empir})
    return out, x_star, lamda_star, idx


def make_sruc_lipschitzness_plot(
    results_sr,
    results_uc,
    x_bars_sr,
    y_bars_sr,
    x_bars_uc,
    initial_data,
    savefig=False,
):

    plt.style.use("/Users/aberk/code/theme_bw.mplstyle")
    plt.rcParams["font.size"] = 20
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["mathtext.fontset"] = "cm"

    A, b = initial_data[:2]
    (
        lipsch_empir_sr,
        x_star_sr,
        lamda_star_sr,
        idx_sr,
    ) = compute_empirical_lipschitzness(results_sr, x_bars_sr)
    lipsch_const_sr = sr.lipschitz_bound_lamda(
        x_star_sr.ravel(),
        A,
        b.ravel(),
        lamda_star_sr,
        y_bars_sr[:, idx_sr].ravel(),
    )
    lipsch_ratio = sr.lipschitz_bound_ratio(
        x_star_sr, A, b, lamda_star_sr, y_bars_sr[:, idx_sr]
    )
    (
        lipsch_empir_uc,
        x_star_uc,
        lamda_star_uc,
        idx_uc,
    ) = compute_empirical_lipschitzness(results_uc, x_bars_uc)
    lipsch_const_uc = uc.lipschitz_bound_lamda(x_star_uc, A, b, lamda_star_uc)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(
        results_sr.lamda / lamda_star_sr,
        lipsch_empir_sr.lipsch_empir,
        "--",
        color="tab:red",
        label=r"$\|\bar{x}_{\rm{SR}}(\lambda) - \bar{x}_{\rm{SR}}(\bar{\lambda}_{\rm{SR}})\|$",
    )
    ax.plot(
        results_sr.lamda / lamda_star_sr,
        lipsch_const_sr * lipsch_empir_sr.dlamda.abs(),
        color="tab:red",
        label=r"$L_{\rm{SR}} |\lambda - \bar{\lambda}_{\rm{SR}}|$",
    )
    ax.plot(
        results_sr.lamda / lamda_star_sr,
        lipsch_const_sr * lipsch_ratio * lipsch_empir_sr.dlamda.abs(),
        color="tab:purple",
        ls="dashdot",
        label=r"$L_{\rm{SR}}|1 - V||\lambda - \bar{\lambda}_{\rm{SR}}|$",
    )
    ax.plot(
        results_uc.lamda / lamda_star_uc,
        lipsch_empir_uc.lipsch_empir,
        "--",
        color="tab:blue",
        label=r"$\|\bar{x}_{\rm{UC}}(\lambda) - \bar{x}_{\rm{UC}}(\bar{\lambda}_{\rm{UC}})\|$",
    )
    ax.plot(
        results_uc.lamda / lamda_star_uc,
        lipsch_const_uc * lipsch_empir_uc.dlamda.abs(),
        color="tab:blue",
        label=r"$L_{\rm{UC}} |\lambda - \bar{\lambda}_{\rm{UC}}|$",
    )
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(-1e-3, 1.5)
    ax.set_xlabel(r"$\lambda_{\rm{nmz}}$")
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    if isinstance(savefig, str):
        fig.savefig(savefig, bbox_inches="tight")
    else:
        plt.show()
    plt.close("all")
    del fig, ax


if __name__ == "__main__":
    m, n, s, gamma = 100, 200, 5, 0.5
    alphas = np.logspace(-1, 1, 101)

    RUN_TRIAL = True
    if RUN_TRIAL:
        (
            results_sr,
            results_uc,
            (x_bars_sr, y_bars_sr, x_bars_uc),
            initial_data,
        ) = run_trial(m, n, s, gamma, alphas=alphas, seed=2023)
    else:
        results_sr, results_uc, solutions, initial_data = load_trial()
        x_bars_sr, y_bars_sr, x_bars_uc = solutions
        initial_data = list(initial_data)

    fig_dir = "fig/example_introduction_sr_uc"
    sr.mkdir(fig_dir)
    savefig = os.path.join(fig_dir, "sr_uc_lipschitz_comparison.pdf")
    make_sruc_lipschitzness_plot(
        results_sr,
        results_uc,
        x_bars_sr,
        y_bars_sr,
        x_bars_uc,
        initial_data,
        savefig,
    )
